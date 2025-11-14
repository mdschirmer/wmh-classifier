#!/usr/bin/env python3
"""WMH Classifier - Continuous and Distance-Based Classification"""

import sys
import argparse
import logging
from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom, binary_dilation
from skimage.measure import label


class WMHClassifier:
    """Classify WMH into zones using continuous or distance-based methods."""
    
    def __init__(self, vent_dilations=1, distance_thresholds=None, 
                 zone_names=None, resample=True, verbose=False):
        self.vent_dilations = vent_dilations
        self.distance_thresholds = distance_thresholds
        self.zone_names = zone_names
        self.resample = resample
        self.method = "continuous" if distance_thresholds is None else "distance"
        
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if self.zone_names is not None:
            expected_zones = len(distance_thresholds) + 1 if distance_thresholds else 2
            if len(self.zone_names) != expected_zones:
                raise ValueError(f"Zone names must match number of zones")
    
    def resample_to_minimum_1mm(self, data, voxel_size):
        """
        Resample to ensure all dimensions are at most 1mm.
        Only upsample dimensions > 1mm, leave others unchanged.
        """
        target_size = tuple(min(vs, 1.0) for vs in voxel_size)
        zoom_factors = tuple(vs / ts for vs, ts in zip(voxel_size, target_size))
        
        if all(zf == 1.0 for zf in zoom_factors):
            self.logger.debug(f"No resampling needed, all dimensions ≤ 1mm")
            return data, target_size
        
        self.logger.debug(f"Original: {voxel_size} mm -> Target: {target_size} mm")
        resampled = zoom(data, zoom_factors, order=0)
        self.logger.debug(f"Shape: {data.shape} -> {resampled.shape}")
        
        return resampled, target_size
    
    def resample_to_original(self, data, target_shape):
        """Resample data back to original resolution."""
        if data.shape == target_shape:
            return data
        
        zoom_factors = tuple(target_shape[i] / data.shape[i] for i in range(3))
        resampled = zoom(data, zoom_factors, order=0)
        return resampled.astype(np.uint8)
    
    def dilate_wmh_inplane(self, mask):
        """
        Dilate WMH by 1 voxel in-plane only (no diagonals, no through-plane).
        Uses a cross-shaped structuring element within each slice.
        """
        # Create cross-shaped structuring element (no diagonals)
        # Shape: 3x3 with only cardinal directions
        struct_2d = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        
        dilated = np.zeros_like(mask)
        
        # Dilate each slice independently
        for slice_idx in range(mask.shape[2]):
            slice_2d = mask[:, :, slice_idx]
            if np.any(slice_2d):  # Only process non-empty slices
                dilated[:, :, slice_idx] = binary_dilation(slice_2d, structure=struct_2d)
        
        self.logger.debug("WMH dilated in-plane (1 voxel, no diagonals)")
        return dilated.astype(np.uint8)
    
    def dilate_mask_3d(self, mask, num_dilations):
        """Dilate mask in 3D."""
        if num_dilations == 0:
            return mask
        
        dilated = mask.copy()
        for iteration in range(num_dilations):
            dilated = binary_dilation(dilated).astype(np.uint8)
        return dilated
    
    def classify_continuous(self, wmh_mask, vent_mask):
        """
        Continuous classification with overlap-based region merging.
        
        Method:
        1. Label all WMH regions
        2. Dilate each region by 1 voxel (6-connected)
        3. Check pairwise: if dilated regions overlap, merge them
        4. Classify final merged regions
        """
        self.logger.debug("Using continuous method with dilated overlap merging")
        
        from scipy.ndimage import binary_dilation, generate_binary_structure
        
        # Initial labeling
        wmh_labels_initial, num_initial = label(wmh_mask, return_num=True, connectivity=3)
        self.logger.debug(f"Initial: {num_initial} separate regions")
        
        if num_initial <= 1:
            wmh_labels = wmh_labels_initial
            num_merged = num_initial
        else:
            # Create 6-connected structuring element (faces only, no diagonals)
            struct = generate_binary_structure(3, 1)
            
            # Dilate each region and store
            dilated_regions = {}
            for label_id in range(1, num_initial + 1):
                region = (wmh_labels_initial == label_id)
                dilated_regions[label_id] = binary_dilation(region, structure=struct)
            
            # Find overlapping pairs by checking dilated regions against each other
            touching_pairs = set()
            labels = list(range(1, num_initial + 1))
            
            for i, label_a in enumerate(labels):
                for label_b in labels[i+1:]:
                    # Check if dilated A overlaps with dilated B
                    if np.any(dilated_regions[label_a] & dilated_regions[label_b]):
                        touching_pairs.add((label_a, label_b))
                        self.logger.debug(f"Dilated regions {label_a} and {label_b} overlap")
            
            self.logger.debug(f"Found {len(touching_pairs)} overlapping pairs")
            
            # Union-find to merge
            parent = list(range(num_initial + 1))
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            for a, b in touching_pairs:
                union(a, b)
            
            # Create merged labels
            label_mapping = {}
            new_label = 0
            for old_label in range(1, num_initial + 1):
                root = find(old_label)
                if root not in label_mapping:
                    new_label += 1
                    label_mapping[root] = new_label
                label_mapping[old_label] = label_mapping[root]
            
            num_merged = new_label
            self.logger.debug(f"After merging: {num_merged} regions")
            
            # Apply merged labels
            wmh_labels = np.zeros_like(wmh_mask, dtype=np.uint8)
            for old_label, new_label_id in label_mapping.items():
                wmh_labels[wmh_labels_initial == old_label] = new_label_id
        
        # Dilate ventricle mask
        self.logger.debug(f"Dilating ventricle mask ({self.vent_dilations} iterations)...")
        vent_dilated = self.dilate_mask_3d(vent_mask, self.vent_dilations)
        
        # Classify merged regions
        output = np.zeros_like(wmh_mask, dtype=np.uint8)
        
        for label_id in range(1, num_merged + 1):
            region = (wmh_labels == label_id)
            if np.any(region & vent_dilated):
                output[region] = 1
            else:
                output[region] = 2
        
        return output
    
    def classify_distance(self, wmh_mask, vent_mask, voxel_size):
        """Distance-based classification."""
        self.logger.debug("Using distance-based method")
        
        # Compute distance transform
        inverted_vent = 1 - vent_mask
        distance_mm = distance_transform_edt(inverted_vent, sampling=voxel_size)
        
        # Create zones
        output = np.zeros_like(wmh_mask, dtype=np.uint8)
        wmh_voxels = wmh_mask > 0
        thresholds = [0] + list(self.distance_thresholds) + [np.inf]
        
        for zone_id in range(len(thresholds) - 1):
            lower, upper = thresholds[zone_id], thresholds[zone_id + 1]
            in_zone = wmh_voxels & (distance_mm >= lower) & (distance_mm < upper)
            output[in_zone] = zone_id + 1
        
        return output
    
    def process_single(self, wmh_path, ventricle_path):
        """Process a single WMH/ventricle pair."""
        self.logger.info(f"Processing: {Path(wmh_path).name}")
        
        # Load images
        wmh_img = nib.load(str(wmh_path))
        vent_img = nib.load(str(ventricle_path))
        
        original_shape = wmh_img.shape
        original_voxel_size = tuple(float(x) for x in wmh_img.header.get_zooms()[:3])
        original_affine = wmh_img.affine
        original_header = wmh_img.header
        
        # Get masks
        wmh_data = (np.asarray(wmh_img.dataobj) > 0).astype(np.uint8)
        vent_data = (np.asarray(vent_img.dataobj) > 0).astype(np.uint8)
        
        # Resample if requested
        if self.resample:
            self.logger.info(f"Checking resolution: {original_voxel_size}")
            wmh_mask, working_voxel_size = self.resample_to_minimum_1mm(wmh_data, original_voxel_size)
            vent_mask, _ = self.resample_to_minimum_1mm(vent_data, original_voxel_size)
            voxel_size = working_voxel_size
        else:
            wmh_mask = wmh_data
            vent_mask = vent_data
            voxel_size = original_voxel_size
        
        self.logger.debug(f"Working resolution: {voxel_size}, shape: {wmh_mask.shape}")
        
        # Classify
        if self.method == "continuous":
            output_mask = self.classify_continuous(wmh_mask, vent_mask)
            method_str = "continuous"
            num_zones = 2
            default_names = ["periventricular", "subcortical"]
        else:
            output_mask = self.classify_distance(wmh_mask, vent_mask, voxel_size)
            method_str = "-".join(map(str, [0] + self.distance_thresholds))
            num_zones = len(self.distance_thresholds) + 1
            
            # Generate zone names
            thresholds = [0] + list(self.distance_thresholds) + [np.inf]
            default_names = []
            for i in range(len(thresholds) - 1):
                lower, upper = thresholds[i], thresholds[i + 1]
                name = f">{lower}mm" if upper == np.inf else f"{lower}-{upper}mm"
                default_names.append(name)
        
        zone_names = self.zone_names if self.zone_names else default_names
        
        # Resample back to original space if needed
        if self.resample and wmh_mask.shape != original_shape:
            self.logger.info("Resampling back to original resolution...")
            output_mask = self.resample_to_original(output_mask, original_shape)
        
        # Calculate statistics using original voxel size
        voxel_volume_mm3 = np.prod(original_voxel_size)
        total_voxels = int(np.sum(output_mask > 0))
        total_volume_cc = voxel_volume_mm3 * total_voxels / 1000
        
        zone_stats = {}
        for zone_id in range(1, num_zones + 1):
            zone_voxels = int(np.sum(output_mask == zone_id))
            zone_volume_cc = voxel_volume_mm3 * zone_voxels / 1000
            zone_percent = (zone_voxels / total_voxels * 100) if total_voxels > 0 else 0
            
            zone_stats[f"zone{zone_id}_voxels"] = zone_voxels
            zone_stats[f"zone{zone_id}_volume_cc"] = float(f"{zone_volume_cc:.4f}")
            zone_stats[f"zone{zone_id}_percent"] = float(f"{zone_percent:.2f}")
            
            self.logger.info(
                f"  Zone {zone_id} ({zone_names[zone_id-1]}): "
                f"{zone_voxels} voxels, {zone_volume_cc:.2f} cc, {zone_percent:.1f}%"
            )
        
        zone_mapping = ";".join([f"{i+1}:{name}" for i, name in enumerate(zone_names)])
        
        result = {
            'wmh_file': str(wmh_path),
            'ventricle_file': str(ventricle_path),
            'method': method_str,
            'zone_mapping': zone_mapping,
            'total_voxels': total_voxels,
            'total_volume_cc': float(f"{total_volume_cc:.4f}"),
            'output_mask': output_mask,
            'affine': original_affine,
            'header': original_header
        }
        result.update(zone_stats)
        
        return result


def save_classified_mask(output_data, affine, header, wmh_path, csv_output_path, method_str):
    """Save classified mask to NIfTI."""
    output_dir = Path(csv_output_path).parent
    wmh_basename = Path(wmh_path).stem.replace('.nii', '')
    
    if method_str == "continuous":
        output_filename = f"{wmh_basename}_wmhc-cont_01.nii.gz"
    else:
        output_filename = f"{wmh_basename}_wmhc-{method_str}_01.nii.gz"
    
    output_path = output_dir / output_filename
    nib.save(nib.Nifti1Image(output_data, affine, header), str(output_path))
    return output_path


def process_batch(classifier, input_csv, output_csv, save_masks=False):
    """Process multiple pairs from CSV."""
    pairs = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        pairs = [{'wmh_mask': row['wmh_mask'], 'ventricle_mask': row['ventricle_mask']} 
                for row in reader]
    
    print(f"Found {len(pairs)} pairs to process")
    results = []
    
    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing...")
        try:
            result = classifier.process_single(pair['wmh_mask'], pair['ventricle_mask'])
            
            if save_masks:
                mask_path = save_classified_mask(
                    result['output_mask'], result['affine'], result['header'],
                    result['wmh_file'], output_csv, result['method']
                )
                result['classified_mask'] = str(mask_path)
            
            result.pop('output_mask', None)
            result.pop('affine', None)
            result.pop('header', None)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            if classifier.logger.level == logging.DEBUG:
                import traceback
                traceback.print_exc()
    
    if results:
        # Get all columns
        all_columns = set()
        for r in results:
            all_columns.update(r.keys())
        
        # Order columns
        base_cols = ['wmh_file', 'ventricle_file', 'method', 'zone_mapping', 
                     'total_voxels', 'total_volume_cc']
        
        # Get zone columns
        zone_nums = set()
        for col in all_columns:
            if col.startswith('zone') and '_' in col:
                zone_num = col.split('_')[0].replace('zone', '')
                if zone_num.isdigit():
                    zone_nums.add(int(zone_num))
        
        zone_cols = []
        for zone_num in sorted(zone_nums):
            zone_cols.extend([
                f"zone{zone_num}_voxels",
                f"zone{zone_num}_volume_cc",
                f"zone{zone_num}_percent"
            ])
        
        other_cols = [c for c in all_columns if c not in base_cols and c not in zone_cols]
        fieldnames = base_cols + zone_cols + other_cols
        
        # Fill missing columns
        for r in results:
            for col in fieldnames:
                if col not in r:
                    r[col] = ''
        
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Processed: {len(results)}/{len(pairs)} pairs")
    else:
        print("\nNo pairs were successfully processed")


def process_single_pair(classifier, wmh_path, vent_path, output_csv, save_masks=False):
    """Process single pair."""
    result = classifier.process_single(wmh_path, vent_path)
    
    if save_masks:
        mask_path = save_classified_mask(
            result['output_mask'], result['affine'], result['header'],
            result['wmh_file'], output_csv, result['method']
        )
        result['classified_mask'] = str(mask_path)
        print(f"\nSaved: {mask_path}")
    
    result.pop('output_mask', None)
    result.pop('affine', None)
    result.pop('header', None)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    
    print(f"\nResults saved to: {output_csv}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WMH Classifier - Spatial stratification of white matter hyperintensities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', help='WMH mask (NIfTI)')
    input_group.add_argument('--input-csv', help='CSV with wmh_mask,ventricle_mask columns')
    
    parser.add_argument('-v', '--ventricle', help='Ventricle mask (required with -i)')
    parser.add_argument('-o', '--output', required=True, help='Output CSV path')
    
    # Classification
    parser.add_argument('--distance-thresholds', 
                       help='Distance thresholds in mm (comma-separated)')
    parser.add_argument('--zone-names', 
                       help='Custom zone names (comma-separated)')
    
    # Ventricle dilation (applies to continuous method only)
    parser.add_argument('--vent-dilation', type=int, default=1,
                       help='Ventricle dilation iterations (continuous method only, default: 1)')
    
    # Resampling
    parser.add_argument('--no-resample', dest='resample', action='store_false',
                       help='Disable resampling (upsamples dimensions >1mm to 1mm)')
    parser.set_defaults(resample=True)
    
    # Output
    parser.add_argument('--save-masks', action='store_true',
                       help='Save classified NIfTI masks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate
    if args.input and not args.ventricle:
        parser.error("-i requires -v")
    
    # Parse thresholds
    distance_thresholds = None
    if args.distance_thresholds:
        try:
            distance_thresholds = [float(x.strip()) 
                                  for x in args.distance_thresholds.split(',')]
            if distance_thresholds != sorted(distance_thresholds):
                parser.error("Thresholds must be in ascending order")
        except ValueError:
            parser.error("Thresholds must be numeric")
    
    # Parse zone names
    zone_names = None
    if args.zone_names:
        zone_names = [x.strip() for x in args.zone_names.split(',')]
    
    # Create classifier
    classifier = WMHClassifier(
        vent_dilations=args.vent_dilation,
        distance_thresholds=distance_thresholds,
        zone_names=zone_names,
        resample=args.resample,
        verbose=args.verbose
    )
    
    # Process
    try:
        if args.input:
            process_single_pair(classifier, args.input, args.ventricle, 
                              args.output, args.save_masks)
        else:
            process_batch(classifier, args.input_csv, args.output, args.save_masks)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()