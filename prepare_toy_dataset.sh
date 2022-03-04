cd toy_data/tomograms/Tomo1/
unzip membranes.zip
wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-10780/map/emd_10780.map.gz
gzip -d emd_10780.map.gz
mv emd_10780.map Tomo1_bin4.map
