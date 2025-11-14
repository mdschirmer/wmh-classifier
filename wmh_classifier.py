#!/usr/bin/env python3
"""WMH Classifier - Continuous and Distance-Based Classification"""

import sys
import argparse
import logging
from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import binary_dilation


class WMHClassifier:
    """Classify WMH into zones using continuous or distance-based methods."""
    
    def __init__(self, wmh_dilations=1, vent_dilations=1, 
                 distance_thresholds=None, zone_names=None, verbose=False):
        self.wmh_dilations = wmh_dilations
        self.vent_dilations = vent_dilations
        self.distance_thresholds = distance_thresholds
        self.zone_names = zone_names
        self.method = "continuous" if distance_thresholds is None else "distance"
        
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if self.zone_names is not None:
            expected_zones = len(distance_thresholds) + 1 if distance_thresholds else 2
            if len(self.zone_names) != expected_zones:
                raise ValueError(f"Zone names must match number of zones")
    
    def dilate_mask(self, mask, num_dilations):
        """Dilate mask slice-by-slice."""
        if num_dilations == 0:
            return mask
        dilated = mask.copy()
        for iteration in range(num_dilations):
            self.logger.debug(f"Dilation iteration {iteration + 1}/{num_dilations}")
            temp = np.zeros_like(dilated)
            for slice_idx in range(dilated.shape[2]):
                temp[:, :, slice_idx] = binary_dilation(dilated[:, :, slice_idx])
            dilated = temp
        return dilated
    
    def classify_continuous(self, wmh_mask, vent_mask):
        """Continuous (dilation-based) classification."""
        self.logger.debug("Using continuous (dilation-based) method")
        
        # Dilate WMH mask
        self.logger.debug(f"Dilating WMH mask ({self.wmh_dilations} iterations)...")
        wmh_dilated = self.dilate_mask(wmh_mask, self.wmh_dilations)
        
        # Label connected components
        self.logger.debug("Labeling WMH regions...")
        wmh_labels, num_labels = label(wmh_dilated, return_num=True, connectivity=1)
        self.logger.debug(f"Found {num_labels} WMH regions")
        
        # Dilate ventricle mask
        self.logger.debug(f"Dilating ventricle mask ({self.vent_dilations} iterations)...")
        vent_dilated = self.dilate_mask(vent_mask, self.vent_dilations)
        
        # Classify each region
        output = np.zeros_like(wmh_mask, dtype=np.uint8)
        for label_id in range(1, num_labels + 1):
            region_mask = (wmh_labels == label_id)
            if np.any(np.logical_and(region_mask, vent_dilated)):
                output[region_mask] = 1  # periventricular
                self.logger.debug(f"Region {label_id}: periventricular")
            else:
                output[region_mask] = 2  # subcortical
                self.logger.debug(f"Region {label_id}: subcortical")
        
        # Constrain to original WMH boundaries
        return output * (wmh_mask > 0)
    
    def classify_distance(self, wmh_mask, vent_mask, voxel_size):
        """Distance-based classification."""
        self.logger.debug("Using distance-based method")
        self.logger.debug(f"Distance thresholds: {self.distance_thresholds} mm")
        
        # Compute distance transform from ventricle surface
        inverted_vent = 1 - vent_mask
        distance_mm = distance_transform_edt(inverted_vent, sampling=voxel_size)
        
        self.logger.debug(f"Distance range: {distance_mm[wmh_mask > 0].min():.2f} - {distance_mm[wmh_mask > 0].max():.2f} mm")
        
        # Create zones based on thresholds
        output = np.zeros_like(wmh_mask, dtype=np.uint8)
        wmh_voxels = wmh_mask > 0
        thresholds = [0] + list(self.distance_thresholds) + [np.inf]
        
        for zone_id in range(len(thresholds) - 1):
            lower, upper = thresholds[zone_id], thresholds[zone_id + 1]
            in_zone = wmh_voxels & (distance_mm >= lower) & (distance_mm < upper)
            output[in_zone] = zone_id + 1
            
            upper_str = f"{upper}mm" if upper != np.inf else "∞"
            self.logger.debug(f"Zone {zone_id + 1} ({lower}-{upper_str}): {np.sum(in_zone)} voxels")
        
        return output
    
    def process_single(self, wmh_path, ventricle_path):
        """Process a single WMH/ventricle pair."""
        self.logger.info(f"Processing: {Path(wmh_path).name}")
        
        # Load images
        wmh_img = nib.load(str(wmh_path))
        vent_img = nib.load(str(ventricle_path))
        wmh_mask = (np.asarray(wmh_img.dataobj) > 0).astype(np.uint8)
        vent_mask = (np.asarray(vent_img.dataobj) > 0).astype(np.uint8)
        
        # Get voxel dimensions
        voxel_size = wmh_img.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_size)
        
        self.logger.debug(f"WMH mask: {wmh_mask.shape}, {np.sum(wmh_mask)} voxels")
        self.logger.debug(f"Ventricle mask: {vent_mask.shape}, {np.sum(vent_mask)} voxels")
        self.logger.debug(f"Voxel size: {voxel_size} mm, volume: {voxel_volume_mm3:.4f} mm³")
        
        # Classify based on method
        if self.method == "continuous":
            output_mask = self.classify_continuous(wmh_mask, vent_mask)
            method_str = "continuous"
            num_zones = 2
            default_names = ["periventricular", "subcortical"]
        else:
            output_mask = self.classify_distance(wmh_mask, vent_mask, voxel_size)
            method_str = "-".join(map(str, [0] + self.distance_thresholds))
            num_zones = len(self.distance_thresholds) + 1
            
            # Auto-generate zone names
            thresholds = [0] + list(self.distance_thresholds) + [np.inf]
            default_names = []
            for i in range(len(thresholds) - 1):
                lower, upper = thresholds[i], thresholds[i + 1]
                if upper == np.inf:
                    default_names.append(f">{lower}mm")
                else:
                    default_names.append(f"{lower}-{upper}mm")
        
        zone_names = self.zone_names if self.zone_names else default_names
        
        # Calculate total
        total_voxels = int(np.sum(wmh_mask))
        total_volume_cc = voxel_volume_mm3 * total_voxels / 1000  # mm³ to cc
        
        # Calculate statistics for each zone
        zone_stats = {}
        for zone_id in range(1, num_zones + 1):
            zone_mask = (output_mask == zone_id)
            zone_voxels = int(np.sum(zone_mask))
            zone_volume_cc = voxel_volume_mm3 * zone_voxels / 1000  # mm³ to cc
            zone_percent = (zone_voxels / total_voxels * 100) if total_voxels > 0 else 0
            
            zone_stats[f"zone{zone_id}_voxels"] = zone_voxels
            zone_stats[f"zone{zone_id}_volume_cc"] = float(f"{zone_volume_cc:.4f}")
            zone_stats[f"zone{zone_id}_percent"] = float(f"{zone_percent:.2f}")
            
            self.logger.info(
                f"  Zone {zone_id} ({zone_names[zone_id-1]}): "
                f"{zone_voxels} voxels, {zone_volume_cc:.2f} cc, {zone_percent:.1f}%"
            )
        
        # Create zone mapping string
        zone_mapping = ";".join([f"{i+1}:{name}" for i, name in enumerate(zone_names)])
        
        # Build result dictionary
        result = {
            'wmh_file': str(wmh_path),
            'ventricle_file': str(ventricle_path),
            'method': method_str,
            'zone_mapping': zone_mapping,
            'total_voxels': total_voxels,
            'total_volume_cc': float(f"{total_volume_cc:.4f}"),
            'output_mask': output_mask,
            'affine': wmh_img.affine,
            'header': wmh_img.header
        }
        
        # Add zone-specific statistics
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
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input CSV
    pairs = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        pairs = [{'wmh_mask': row['wmh_mask'], 'ventricle_mask': row['ventricle_mask']} for row in reader]
    
    print(f"Found {len(pairs)} pairs to process")
    
    # Process each pair
    results = []
    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing pair...")
        try:
            result = classifier.process_single(pair['wmh_mask'], pair['ventricle_mask'])
            
            # Save mask if requested
            if save_masks:
                mask_path = save_classified_mask(
                    result['output_mask'], result['affine'], result['header'],
                    result['wmh_file'], output_csv, result['method']
                )
                result['classified_mask'] = str(mask_path)
                print(f"  Saved: {mask_path.name}")
            
            # Remove mask data from result
            result.pop('output_mask', None)
            result.pop('affine', None)
            result.pop('header', None)
            
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            if classifier.logger.level == logging.DEBUG:
                import traceback
                traceback.print_exc()
    
    # Write output CSV with dynamic columns
    if results:
        # Determine column order
        all_columns = set()
        for r in results:
            all_columns.update(r.keys())
        
        base_cols = ['wmh_file', 'ventricle_file', 'method', 'zone_mapping', 
                     'total_voxels', 'total_volume_cc']
        
        # Get zone columns in order (voxels, volume, percent for each zone)
        zone_nums = set()
        for col in all_columns:
            if col.startswith('zone') and '_' in col:
                zone_num = col.split('_')[0].replace('zone', '')
                if zone_num.isdigit():
                    zone_nums.add(int(zone_num))
        
        zone_cols = []
        for zone_num in sorted(zone_nums):
            zone_cols.append(f"zone{zone_num}_voxels")
            zone_cols.append(f"zone{zone_num}_volume_cc")
            zone_cols.append(f"zone{zone_num}_percent")
        
        # Other columns
        other_cols = [c for c in all_columns if c not in base_cols and c not in zone_cols]
        
        fieldnames = base_cols + zone_cols + other_cols
        
        # Fill missing columns
        for r in results:
            for col in fieldnames:
                if col not in r:
                    r[col] = ''
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_csv}")
        print(f"Processed: {len(results)}/{len(pairs)} pairs successfully")
        print(f"{'='*60}")
    else:
        print("\nNo pairs were successfully processed")


def process_single_pair(classifier, wmh_path, vent_path, output_csv, save_masks=False):
    """Process single pair."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    result = classifier.process_single(wmh_path, vent_path)
    
    if save_masks:
        mask_path = save_classified_mask(
            result['output_mask'], result['affine'], result['header'],
            result['wmh_file'], output_csv, result['method']
        )
        result['classified_mask'] = str(mask_path)
        print(f"\nSaved classified mask: {mask_path}")
    
    result.pop('output_mask', None)
    result.pop('affine', None)
    result.pop('header', None)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WMH Classifier - Spatial stratification of white matter hyperintensities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuous method (default)
  %(prog)s -i wmh.nii.gz -v vent.nii.gz -o stats.csv
  
  # 10mm distance threshold
  %(prog)s -i wmh.nii.gz -v vent.nii.gz -o stats.csv --distance-thresholds 10
  
  # Juxta method with zone names
  %(prog)s -i wmh.nii.gz -v vent.nii.gz -o stats.csv \\
      --distance-thresholds 3,13 --zone-names juxta,peri,deep
  
  # Batch processing
  %(prog)s --input-csv pairs.csv -o results.csv --distance-thresholds 3,13 --save-masks
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', help='WMH mask (NIfTI)')
    input_group.add_argument('--input-csv', help='CSV with wmh_mask,ventricle_mask columns')
    
    parser.add_argument('-v', '--ventricle', help='Ventricle mask (NIfTI, required with -i)')
    parser.add_argument('-o', '--output', required=True, help='Output CSV path')
    
    # Classification
    parser.add_argument('--distance-thresholds', 
                       help='Distance thresholds in mm (comma-separated). E.g., "10" or "3,13"')
    parser.add_argument('--zone-names', 
                       help='Custom zone names (comma-separated). Must match number of zones')
    
    # Continuous method parameters
    parser.add_argument('--wmh-dilation', type=int, default=1,
                       help='WMH dilation iterations (continuous method only, default: 1)')
    parser.add_argument('--vent-dilation', type=int, default=1,
                       help='Ventricle dilation iterations (continuous method only, default: 1)')
    
    # Output
    parser.add_argument('--save-masks', action='store_true',
                       help='Save classified NIfTI masks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate
    if args.input and not args.ventricle:
        parser.error("-i/--input requires -v/--ventricle")
    
    # Parse distance thresholds
    distance_thresholds = None
    if args.distance_thresholds:
        try:
            distance_thresholds = [float(x.strip()) for x in args.distance_thresholds.split(',')]
            if distance_thresholds != sorted(distance_thresholds):
                parser.error("Distance thresholds must be in ascending order")
        except ValueError:
            parser.error("Distance thresholds must be numeric")
    
    # Parse zone names
    zone_names = None
    if args.zone_names:
        zone_names = [x.strip() for x in args.zone_names.split(',')]
    
    # Initialize classifier
    classifier = WMHClassifier(
        wmh_dilations=args.wmh_dilation,
        vent_dilations=args.vent_dilation,
        distance_thresholds=distance_thresholds,
        zone_names=zone_names,
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