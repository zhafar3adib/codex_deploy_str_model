"""Monthly training, using the same RF grid and split as the updated Colab."""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from model_utils import FEATURES, TARGETS, dropdowns, encode, prepare_data

PARAM_GRID = {
    'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
}


def train_bundle(raw, as_of, param_grid=None):
    data = prepare_data(raw, as_of)
    if len(data) < 20:
        raise ValueError('Minimal 20 artikel valid berusia >=90 hari diperlukan untuk split dan CV.')
    # Preserve Colab's encoder fit on eligible df_reg, followed by 70/30 split.
    encoder = OneHotEncoder(sparse_output=False)
    features = encoder.fit_transform(data[FEATURES])
    X = pd.DataFrame(features, columns=encoder.get_feature_names_out(FEATURES))
    y = data[TARGETS].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=43),
                          param_grid=PARAM_GRID if param_grid is None else param_grid,
                          cv=5, scoring='r2', n_jobs=-1, error_score='raise')
    grid_rf.fit(X_train, y_train)
    best_rf_mod = grid_rf.best_estimator_
    prediction = best_rf_mod.predict(X_test)
    if not np.isfinite(prediction).all() or not np.isfinite(grid_rf.best_score_):
        raise ValueError('Evaluasi model menghasilkan nilai tidak finite; model tidak diterbitkan.')
    metrics = {name: {
        'mae': float(mean_absolute_error(y_test[name], prediction[:, i])),
        'mse': float(mean_squared_error(y_test[name], prediction[:, i])),
        'r2': float(r2_score(y_test[name], prediction[:, i])),
    } for i, name in enumerate(TARGETS)}
    now = datetime.now(ZoneInfo('Asia/Bangkok'))
    bundle = {
        'schema_version': 2, 'model_name': 'best_rf_mod', 'model': best_rf_mod,
        'encoder': encoder, 'categorical_columns': FEATURES, 'target_columns': TARGETS,
        'feature_columns': list(X.columns), 'dropdown_options': dropdowns(data),
        # Holdout articles must never appear in the similarity table.
        'training_articles': data.loc[X_train.index, ['artikel', 'tanggal'] + FEATURES].reset_index(drop=True),
        'metadata': {
            'version': now.strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8],
            'trained_at': now.isoformat(), 'as_of': str(as_of),
            'minimum_age_days': 90, 'eligible_rows': len(data),
            'training_rows': len(X_train), 'test_rows': len(X_test),
            'best_params': grid_rf.best_params_, 'cv_r2': float(grid_rf.best_score_),
            'metrics': metrics,
            'library_versions': {p: version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'scipy', 'joblib']},
        },
    }
    np.testing.assert_allclose(best_rf_mod.predict(encode(bundle, data.loc[X_test.index])), prediction)
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='klamby-469003')
    parser.add_argument('--bucket', default='klamby_sales')
    parser.add_argument('--output', default='work/training')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    # Include Drive scope because the existing BigQuery source can use external Sheets.
    import google.auth
    from google.cloud import bigquery
    scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/drive']
    credentials, _ = google.auth.default(scopes=scopes)
    as_of = datetime.now(ZoneInfo('Asia/Bangkok')).date()
    client = bigquery.Client(project=args.project, credentials=credentials)
    query = Path(__file__).with_name('training.sql').read_text(encoding='utf-8')
    config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('as_of', 'DATE', as_of)])
    raw = client.query(query, job_config=config).to_dataframe(create_bqstorage_client=False)
    bundle = train_bundle(raw, as_of)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'model_bundle.joblib'
    joblib.dump(bundle, path, compress=3)
    # Verify serialized bundle before publishing the active pointer.
    restored = joblib.load(path)
    sample = restored['training_articles'].iloc[[0]]
    np.testing.assert_allclose(restored['model'].predict(encode(restored, sample)),
                               bundle['model'].predict(encode(bundle, sample)))
    (output / 'metrics.json').write_text(json.dumps(bundle['metadata'], indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(bundle['metadata'], indent=2, allow_nan=False))
    if args.publish:
        from model_store import publish_bundle
        from google.cloud import storage
        publish_bundle(storage.Client(project=args.project, credentials=credentials), args.bucket, path, bundle['metadata'])
        print('Model, encoder, dropdown, dan artikel training berhasil diterbitkan.')


if __name__ == '__main__':
    main()
