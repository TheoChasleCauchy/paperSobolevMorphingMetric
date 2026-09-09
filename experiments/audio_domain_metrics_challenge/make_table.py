import os, csv

# Make a table of the mean and standard deviation result of each metric
def make_table(results_dir):
    import re
    
    def get_metrics_values(results_dir, no_audio: bool = False):
        metrics_values = {}
        
        # Get Smoothness Clap metric value
        clap_smoothness_clap_csv_path = os.path.join(results_dir, "LaionCLAP_audio", "LaionCLAP_audio_smoothness_MF_values.csv")
        with open(clap_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["CLAP Smoothness MF"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)

        mert_smoothness_clap_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_MF_values.csv")
        with open(mert_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["MERT Smoothness MF"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
        # Get Sobolev k=0, p=2 value
        sobolev_k0_p2_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_sobolev_dists_0_2.csv")
        with open(sobolev_k0_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k0_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k0_p2, std_sobolev_k0_p2 = map(float, mean_std_sobolev_k0_p2)
            metrics_values["Sobolev (0, 2)"] = (mean_sobolev_k0_p2, std_sobolev_k0_p2)
        
        # Get Sobolev k=1, p=2 value
        sobolev_k1_p2_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_sobolev_dists_1_2.csv")
        with open(sobolev_k1_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k1_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k1_p2, std_sobolev_k1_p2 = map(float, mean_std_sobolev_k1_p2)
            metrics_values["Sobolev (1, 2)"] = (mean_sobolev_k1_p2, std_sobolev_k1_p2)
        
        # Get Correspondence value
        if no_audio:
            metrics_values["MFCC Correspondence SM"] = (404.0, 404.0)
        else:
            mfcc_correspondence_csv_path = os.path.join(results_dir, "MFCC", "MFCC_correspondence_SM_values.csv")
            with open(mfcc_correspondence_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
                metrics_values["MFCC Correspondence SM"] = (mean_correspondence, std_correspondence)

        mert_correspondence_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_correspondence_SM_values.csv")
        with open(mert_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MERT Correspondence SM"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness value
        if no_audio:
            metrics_values["CDPAM Intermediateness SM"] = (404.0, 404.0)
        else:
            cdpam_intermediateness_csv_path = os.path.join(results_dir, "CDPAM", "CDPAM_intermediateness_SM_values.csv")
            with open(cdpam_intermediateness_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
                metrics_values["CDPAM Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        mert_intermediateness_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_intermediateness_SM_values.csv")
        with open(mert_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["MERT Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        # Get Smoothness CDPAM value
        if no_audio:
            metrics_values["CDPAM Smoothness SM"] = (404.0, 404.0)
        else:
            cdpam_smoothness_cdpam_csv_path = os.path.join(results_dir, "CDPAM", "CDPAM_smoothness_SM_values.csv")
            with open(cdpam_smoothness_cdpam_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
                metrics_values["CDPAM Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)

        mert_smoothness_cdpam_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_SM_values.csv")
        with open(mert_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["MERT Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
    
    results_dir_ref = f"{results_dir}/results_ref_trajectories"
    ref_metrics_values = get_metrics_values(results_dir_ref)
    results_dir_null = f"{results_dir}/results_null_trajectories"
    null_metrics_values = get_metrics_values(results_dir_null)
        
    results_dir_nuc = f"{results_dir}/results_nuc_trajectories"
    nuc_metrics_values = get_metrics_values(results_dir_nuc)
    results_dir_eqc = f"{results_dir}/results_eqc_trajectories"
    eqc_metrics_values = get_metrics_values(results_dir_eqc, no_audio=True)

    # Write the table to a CSV file
    output_csv_path = os.path.join(results_dir, "audio_domain_metrics_values_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: metrics as rows
        header = ["Metric", "Encoder", "Ref", "NUC", "EQC",  "Null"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "Correspondence", 
            "MFCC",
            f"{ref_metrics_values['MFCC Correspondence SM'][0]:.2f} ({ref_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{nuc_metrics_values['MFCC Correspondence SM'][0]:.2f} ({nuc_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{eqc_metrics_values['MFCC Correspondence SM'][0]:.2f} ({eqc_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{null_metrics_values['MFCC Correspondence SM'][0]:.2f} ({null_metrics_values['MFCC Correspondence SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "L-CLAP audio",
            f"{ref_metrics_values['CLAP Smoothness MF'][0]:.2f} ({ref_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{nuc_metrics_values['CLAP Smoothness MF'][0]:.2f} ({nuc_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{eqc_metrics_values['CLAP Smoothness MF'][0]:.2f} ({eqc_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{null_metrics_values['CLAP Smoothness MF'][0]:.2f} ({null_metrics_values['CLAP Smoothness MF'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "CDPAM",
            f"{ref_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({ref_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{nuc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({nuc_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{eqc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({eqc_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{null_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({null_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            "CDPAM",
            f"{ref_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({ref_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{nuc_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({nuc_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{eqc_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({eqc_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{null_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({null_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
        ]
        writer.writerow(row)

        # MERT
        row = [
            "Correspondence", 
            "MERT",
            f"{ref_metrics_values['MERT Correspondence SM'][0]:.2f} ({ref_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Correspondence SM'][0]:.2f} ({nuc_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Correspondence SM'][0]:.2f} ({eqc_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{null_metrics_values['MERT Correspondence SM'][0]:.2f} ({null_metrics_values['MERT Correspondence SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "MERT",
            f"{ref_metrics_values['MERT Smoothness MF'][0]:.2f} ({ref_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{nuc_metrics_values['MERT Smoothness MF'][0]:.2f} ({nuc_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{eqc_metrics_values['MERT Smoothness MF'][0]:.2f} ({eqc_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{null_metrics_values['MERT Smoothness MF'][0]:.2f} ({null_metrics_values['MERT Smoothness MF'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "MERT",
            f"{ref_metrics_values['MERT Intermediateness SM'][0]:.2f} ({ref_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Intermediateness SM'][0]:.2f} ({nuc_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Intermediateness SM'][0]:.2f} ({eqc_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{null_metrics_values['MERT Intermediateness SM'][0]:.2f} ({null_metrics_values['MERT Intermediateness SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            "MERT",
            f"{ref_metrics_values['MERT Smoothness SM'][0]:.2f} ({ref_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Smoothness SM'][0]:.2f} ({nuc_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Smoothness SM'][0]:.2f} ({eqc_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{null_metrics_values['MERT Smoothness SM'][0]:.2f} ({null_metrics_values['MERT Smoothness SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            "MERT",
            f"{ref_metrics_values['Sobolev (0, 2)'][0]:.2f} ({ref_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{nuc_metrics_values['Sobolev (0, 2)'][0]:.2f} ({nuc_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{eqc_metrics_values['Sobolev (0, 2)'][0]:.2f} ({eqc_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{null_metrics_values['Sobolev (0, 2)'][0]:.2f} ({null_metrics_values['Sobolev (0, 2)'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            "MERT",
            f"{ref_metrics_values['Sobolev (1, 2)'][0]:.2f} ({ref_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{nuc_metrics_values['Sobolev (1, 2)'][0]:.2f} ({nuc_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{eqc_metrics_values['Sobolev (1, 2)'][0]:.2f} ({eqc_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{null_metrics_values['Sobolev (1, 2)'][0]:.2f} ({null_metrics_values['Sobolev (1, 2)'][1]:.2f})",
        ]
        writer.writerow(row)

    print(f"Table generated successfully at {output_csv_path}.")