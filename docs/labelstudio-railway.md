# Label Studio on Railway — IFS human-rating study (runbook)

Hosts Label Studio with a shareable link so Burmese raters score translations.
Persistent Postgres + volume so collected ratings survive redeploys. ~$5/mo (trial
credit covers the eval window). Artifacts are already built:

- import tasks: `experiments/results/ratings_ls_import.csv` (160 blind tasks: id, src_en, hyp_my)
- labeling UI: `experiments/results/ratings_ls_config.xml`
- private key (NOT uploaded): `experiments/results/ratings_key.csv` (id → system, reference)
- export → analysis: `scripts/ls_export_to_csv.py` → `src/eval/correlate.py`

## 1. Create the Label Studio service

1. railway.app → **New Project** → **Deploy a Docker Image** → image: `heartexlabs/label-studio:latest`.
   - (Or check **Railway Templates** for "Label Studio" for a one-click that bundles Postgres.)
2. Service → **Settings → Deploy → Custom Start Command**:
   `label-studio start --host 0.0.0.0 --port $PORT`

## 2. Add Postgres (so ratings persist)

- Project → **New → Database → PostgreSQL**.

## 3. Add a persistent volume (uploads/media)

- Label Studio service → **Variables/Settings → Volumes** → mount at `/label-studio/data`.

## 4. Set the service variables (use Railway ${{Postgres.*}} references)

```
DJANGO_DB=default
POSTGRE_NAME=${{Postgres.PGDATABASE}}
POSTGRE_USER=${{Postgres.PGUSER}}
POSTGRE_PASSWORD=${{Postgres.PGPASSWORD}}
POSTGRE_HOST=${{Postgres.PGHOST}}
POSTGRE_PORT=${{Postgres.PGPORT}}
LABEL_STUDIO_HOST=https://${{RAILWAY_PUBLIC_DOMAIN}}
CSRF_TRUSTED_ORIGINS=https://${{RAILWAY_PUBLIC_DOMAIN}}
# optional initial admin:
LABEL_STUDIO_USERNAME=you@example.com
LABEL_STUDIO_PASSWORD=<choose-a-strong-one>
```

(Env var names are `POSTGRE_*`, not `POSTGRES_*` — Label Studio's spelling.)

## 5. Generate the public domain

- Service → **Settings → Networking → Generate Domain** → redeploy so
  `LABEL_STUDIO_HOST` / `CSRF_TRUSTED_ORIGINS` pick it up.

## 6. Set up the project

1. Open the domain → sign in (the admin you set) → **Create Project** "WikiHow-MY IFS".
2. **Settings → Labeling Interface → Code** → paste the contents of `ratings_ls_config.xml`.
3. **Data Import** → upload `ratings_ls_import.csv` → import as **160 tasks** (each CSV column
   becomes a field: `$src_en`, `$hyp_my`, `$id`). Confirm one task shows source + translation +
   the two 1–5 rating widgets.
4. **Do NOT upload `ratings_key.csv`** (it holds the system + reference — keeps the study blind).

## 7. Recruit + share

- Add annotator accounts (Organization → People → invite) or enable a shared labeling link.
- Send to Burmese raters (SpeakProof / FB / Telegram). Give the **first ~50 tasks to 2–3 raters**
  (overlap subset) so we can compute Krippendorff's α. Consider 2–3 attention-check items.

## 8. Collect → analyze

```
# Export from Label Studio: Export → JSON → save as experiments/results/ls_export.json
python scripts/ls_export_to_csv.py experiments/results/ls_export.json
python src/eval/correlate.py --ratings experiments/results/ratings_filled.csv \
    --key experiments/results/ratings_key.csv
python src/eval/make_tables.py    # writes paper/tables/ifs_correlation.tex
```

That produces the IFS-vs-human-followability correlation + Williams significance — the paper's
validation of IFS.

## Notes / gotchas

- Verify env-var names against the Label Studio version you deploy (they evolve); the `POSTGRE_*`
  set above matches recent images.
- If links/CSRF errors appear, double-check `LABEL_STUDIO_HOST` + `CSRF_TRUSTED_ORIGINS` match the
  generated domain (https, no trailing slash).
- Keep the eval to this 160-item subset (not the full corpus) so English source exposure stays minimal.
- Delete/pause the Railway service when the eval is done to stop billing.
