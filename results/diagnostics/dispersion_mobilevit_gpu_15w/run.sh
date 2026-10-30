R="vit_mobilevit_deploy_fp32 vit_mobilevit_deploy_per_channel_asymmetric_int8"
D=results/diagnostics/dispersion_mobilevit_gpu_15w
B=$(ls -d results/backups/antes_gpu15w_* | tail -1)
python scripts/dispersion_rows.py snapshot --device-id jetson_gpu_15w --out-dir $D --rep 1 --rows $R --from-backup $B
for rep in 2 3; do
  python -m scripts.run_batch --manifest $D/manifest.csv --branch vit --device-id jetson_gpu_15w \
    --with-power --temp-cooldown --force --repeats 5000
  python scripts/dispersion_rows.py snapshot --device-id jetson_gpu_15w --out-dir $D --rep $rep --rows $R
done
python scripts/dispersion_rows.py finalize --device-id jetson_gpu_15w --out-dir $D --rows $R
