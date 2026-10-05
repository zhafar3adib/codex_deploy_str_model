# Jalankan pada notebook revisi setelah grid_rf.fit selesai.
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from zoneinfo import ZoneInfo
from uuid import uuid4
import json
import platform
import shutil
import joblib
import numpy as np
import pandas as pd
from google.colab import files

if 'df_articles' not in globals():
    raise ValueError('Jalankan ulang cell cleaning notebook revisi untuk menyimpan artikel dan tanggal launching.')
cutoff = pd.Timestamp(datetime.now(ZoneInfo('Asia/Bangkok')).date()) - pd.Timedelta(days=90)
if not (pd.to_datetime(df_articles['tanggal']) <= cutoff).all():
    raise ValueError('Data masih memuat produk <90 hari. Jalankan ulang cleaning dan training.')
if df_articles['artikel'].duplicated().any():
    raise ValueError('Artikel duplikat. Periksa data sumber sebelum training.')
cat_cols = list(categorical_columns)
input_df = df_reg[cat_cols].astype(object).where(df_reg[cat_cols].notna(), 0).astype(str)
dropdown_options = {c: input_df[c].unique().tolist() for c in cat_cols}
dropdown_options['bulan'] = sorted(dropdown_options['bulan'], key=int)
sample_values = encoder.transform(input_df.loc[X_train.index[:1]])
sample_x = pd.DataFrame(sample_values, columns=encoder.get_feature_names_out(cat_cols))[X.columns]
np.testing.assert_allclose(best_rf_mod.predict(sample_x), best_rf_mod.predict(X_train.iloc[[0]]))
references = df_articles.loc[X_train.index, ['artikel', 'tanggal']].copy()
references[cat_cols] = input_df.loc[X_train.index]
now = datetime.now(ZoneInfo('Asia/Bangkok'))
metadata = {
    'version': now.strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8],
    'trained_at': now.isoformat(), 'as_of': str(now.date()), 'minimum_age_days': 90,
    'eligible_rows': len(df_reg), 'training_rows': len(X_train), 'test_rows': len(X_test),
    'best_params': grid_rf.best_params_,
    'library_versions': {p: version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'scipy', 'joblib']},
}
folder = Path('/content/str_model_export')
folder.mkdir(exist_ok=True)
joblib.dump({
    'schema_version': 2, 'model_name': 'best_rf_mod', 'model': best_rf_mod,
    'encoder': encoder, 'categorical_columns': cat_cols, 'feature_columns': list(X.columns),
    'target_columns': list(y.columns), 'dropdown_options': dropdown_options,
    'training_articles': references.reset_index(drop=True), 'metadata': metadata,
}, folder / 'model_bundle.joblib', compress=3)
requirements = ['streamlit>=1.40,<2', 'google-cloud-storage>=3,<4']
requirements += [f'{p}=={v}' for p, v in metadata['library_versions'].items()]
(folder / 'requirements.txt').write_text('\n'.join(requirements) + '\n')
(folder / 'python-version.txt').write_text(platform.python_version() + '\n')
(folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
files.download(shutil.make_archive('/content/str_model_export', 'zip', folder))
