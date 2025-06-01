"""
Script to clean and annotate a phylogenetic tree with MSA data.

Outputs:
- MSA_80_cluster_cleaned.treefile: cleaned tree with leaf names
- dataset_colorstrip.txt: annotation file for top 5 most frequent common names
- dataset_colorstrip_all.txt: annotation file for all unique common names
"""
import re
import pandas as pd

# Clean up tree names
def convert_name(name):
    """Convert a name like 'Acanthaster_planci_3' to 'AcanthasterPlanci3'."""
    # Remove all single quotes from the string
    name = name.replace("'", "")
    
    parts = name.split('_')
    return ''.join([p.capitalize() for p in parts[:-1]]) + parts[-1]

with open("MSA_80cluster.treefile", "r") as f:
    tree = f.read()
leaf_names = set(re.findall(r"[\"']?[\w\.\-]+_\d+[\"']?", tree))
for name in sorted(leaf_names, key=len, reverse=True):
    new_name = convert_name(name)
    tree = tree.replace(name, new_name)
with open("MSA_80cluster_cleaned.treefile", "w") as f:
    f.write(tree)



# Generate annotations for MSA data by 'Common name'
msa_df = pd.read_csv("MSA_80cluster_with_blast_names.csv")
msa_df["Formatted taxon"] = msa_df["Common_taxon"].apply(convert_name)
top_common_names = msa_df["Common name"].value_counts().nlargest(5).index.tolist()

fixed_colors = ['#226e9c', '#AF588A', '#E64B35', '#F6921E', '#228B3B']
top5_color_map = dict(zip(top_common_names, fixed_colors))

itol_top5_lines = []
for _, row in msa_df.iterrows():
    name = row["Common name"]
    if name in top5_color_map:
        line = f"{row['Formatted taxon']}\t{top5_color_map[name]}"
        itol_top5_lines.append(line)

with open("dataset_colorstrip.txt", "w") as f:
    f.write("DATASET_COLORSTRIP\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tCladeHighlights\n")
    f.write("COLOR\t#000000\n\n")
    f.write("STRIP_WIDTH\t100\n")
    f.write("MARGIN\t5\n")
    f.write("BORDER_WIDTH\t1\n\n")
    f.write("LEGEND_TITLE\tTop Common Name Clades\n")
    f.write("LEGEND_SHAPES\t" + "\t".join(['1'] * len(top5_color_map)) + "\n")
    f.write("LEGEND_COLORS\t" + "\t".join(top5_color_map.values()) + "\n")
    f.write("LEGEND_LABELS\t" + "\t".join(top5_color_map.keys()) + "\n\n")
    f.write("DATA\n")
    f.write("\n".join(itol_top5_lines))



# Generate annotations for MSA data for all classes
import seaborn as sns

# Sort common names by frequency
value_counts = msa_df["Common name"].value_counts()
sorted_common_names = value_counts.index.tolist()
top_common_names = sorted_common_names[:5]
fixed_colors = ['#226e9c', '#AF588A', '#E64B35', '#F6921E', '#228B3B']
color_map = dict(zip(top_common_names, fixed_colors))
remaining_names = sorted_common_names[5:]
remaining_colors = sns.color_palette("hls", len(remaining_names)).as_hex()
color_map.update(dict(zip(remaining_names, remaining_colors)))

itol_all_lines = []
for name in sorted_common_names:
    rows = msa_df[msa_df["Common name"] == name]
    for _, row in rows.iterrows():
        line = f"{row['Formatted taxon']}\t{color_map[name]}"
        itol_all_lines.append(line)
        
with open("dataset_colorstrip_all.txt", "w") as f:
    f.write("DATASET_COLORSTRIP\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tAllClades\n")
    f.write("COLOR\t#000000\n\n")
    f.write("STRIP_WIDTH\t100\n")
    f.write("MARGIN\t5\n")
    f.write("BORDER_WIDTH\t1\n\n")

    f.write("LEGEND_TITLE\tAll Common Names (by frequency)\n")
    f.write("LEGEND_SHAPES\t" + "\t".join(['1'] * len(sorted_common_names)) + "\n")
    f.write("LEGEND_COLORS\t" + "\t".join([color_map[name] for name in sorted_common_names]) + "\n")
    f.write("LEGEND_LABELS\t" + "\t".join(sorted_common_names) + "\n\n")

    f.write("DATA\n")
    f.write("\n".join(itol_all_lines))



# Report percentage of fish and nematode taxa
fish_taxa = msa_df[msa_df["Common name"].str.contains("Fish", case=False, na=False)]
nematode_taxa = msa_df[msa_df["Common name"].str.contains("Nematode", case=False, na=False)]
fish_taxa_count = fish_taxa.shape[0]
nematode_taxa_count = nematode_taxa.shape[0]
fish_taxa_percent = fish_taxa_count / msa_df.shape[0] * 100
nematode_taxa_percent = nematode_taxa_count / msa_df.shape[0] * 100

print(f"Percentage of fish taxa: {fish_taxa_percent:.2f}%")
print(f"Percentage of nematode taxa: {nematode_taxa_percent:.2f}%")