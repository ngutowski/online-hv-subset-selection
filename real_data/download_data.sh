#!/usr/bin/env bash
# Downloads the public datasets used in the real-data case study into ./data/
# (Criteo must be downloaded manually, see the end of this script.)
# Data files are NOT redistributed in this repository (MovieLens license forbids redistribution).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/kuairand-pure data/ml-1m data/goodbooks-10k

# --- MovieLens-1M (GroupLens, official) -------------------------------------
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -o -j data/ml-1m.zip 'ml-1m/ratings.dat' 'ml-1m/users.dat' 'ml-1m/movies.dat' -d data/ml-1m
rm data/ml-1m.zip

# --- Goodbooks-10k (Zajac, GitHub) ------------------------------------------
GB=https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master
for f in ratings.csv to_read.csv books.csv; do curl -L -o data/goodbooks-10k/$f $GB/$f; done

# --- KuaiRand-Pure (Gao et al., CIKM 2022) ----------------------------------
# Official download: https://kuairand.com  (KuaiRand-Pure archive). The two log files used are
#   log_standard_4_08_to_4_21_pure.csv and log_standard_4_22_to_5_08_pure.csv
# Copy them into data/kuairand-pure/. The lines below fetch them from the public KuaiSim mirror.
KS=https://raw.githubusercontent.com/Applied-Machine-Learning-Lab/KuaiSim/main/dataset/kuairand/kuairand-Pure/data
for f in log_standard_4_08_to_4_21_pure.csv log_standard_4_22_to_5_08_pure.csv; do curl -L -o data/kuairand-pure/$f $KS/$f; done
# --- ASlib SAT12 and QBF-2014 scenarios (Bischl et al., 2016) ----------------
mkdir -p data/aslib
for sc in SAT12-ALL SAT12-HAND SAT12-INDU SAT12-RAND QBF-2014; do
  mkdir -p data/aslib/$sc
  curl -L -o data/aslib/$sc/algorithm_runs.arff https://raw.githubusercontent.com/coseal/aslib_data/master/$sc/algorithm_runs.arff
done

# --- Criteo Attribution Modeling for Bidding (Diemert et al., 2017) ---------
# Not downloaded automatically (license acceptance / Kaggle login). Get pcb_dataset_final.tsv from
#   https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/
#   or https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling
# then reduce it (~2.4 GB -> ~10 MB) with:
#   python3 reduce_criteo.py /path/to/pcb_dataset_final.tsv
# which writes data/criteo/criteo_agg.csv.gz.
if [ ! -f data/criteo/criteo_agg.csv.gz ]; then
  echo "NOTE: data/criteo/criteo_agg.csv.gz missing -> see the Criteo instructions at the end of download_data.sh"
fi
echo "Done."
