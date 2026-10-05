from datetime import date
import hashlib
import io
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from model_utils import FEATURES, CHARACTERISTICS, dropdowns, encode, month_options, prepare_data, similar_articles
from train import train_bundle
from model_store import MANIFEST, download_bundle, publish_bundle


def raw_data():
    rows = []
    for i in range(40):
        rows.append({
            'artikel': f'ART-{i:03}', 'tanggal': pd.Timestamp('2025-01-01') + pd.Timedelta(days=i * 9),
            **{c: 'A' if i % 2 else 'B' for c in CHARACTERISTICS},
            'design_border': None if i % 3 else 'BORDER',
            'stock': 100 + i * 10, 'quantity30': 20 + i, 'quantity60': 40 + i,
            'quantity90': 60 + i,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def bundle():
    return train_bundle(raw_data(), date(2026, 10, 5),
                        {'n_estimators': [5], 'max_depth': [3], 'min_samples_leaf': [1]})


def test_age_boundary_and_null_zero():
    raw = raw_data().iloc[:5].copy()
    today = pd.Timestamp('2026-10-05')
    raw['tanggal'] = [today - pd.Timedelta(days=89), today - pd.Timedelta(days=90),
                      today - pd.Timedelta(days=91), pd.NaT, today + pd.Timedelta(days=1)]
    data = prepare_data(raw, today.date())
    assert data['artikel'].tolist() == ['ART-001', 'ART-002']
    assert data['design_border'].tolist() == ['0', '0']
    assert dropdowns(data)['design_border'] == ['0']


def test_months_sorted_and_observed_only():
    assert month_options(['10', '2', '12', '1', '2']) == ['1', '2', '10', '12']


def test_invalid_stock_and_duplicate_articles():
    raw = raw_data()
    raw.loc[0, 'stock'] = 0
    raw.loc[1, 'stock'] = np.nan
    assert len(prepare_data(raw, date(2026, 10, 5))) == 38
    raw.loc[3, 'artikel'] = raw.loc[2, 'artikel']
    with pytest.raises(ValueError, match='duplikat'):
        prepare_data(raw, date(2026, 10, 5))


def test_similarity_threshold_date_order_month_and_zero():
    inputs = dict.fromkeys(FEATURES, 'A')
    rows = []
    for article, matching, launched in [('old-high', 9, '2025-01-01'),
                                       ('new-low', 5, '2025-10-01'), ('excluded', 4, '2025-12-01')]:
        rows.append({'artikel': article, 'tanggal': launched, 'bulan': '12',
                     **{c: 'A' if i < matching else 'B' for i, c in enumerate(CHARACTERISTICS)}})
    found = similar_articles(pd.DataFrame(rows), inputs)
    assert found['artikel'].tolist() == ['new-low', 'old-high']
    assert found['jumlah_sama'].tolist() == [5, 9]
    assert found['skor_kemiripan'].tolist() == pytest.approx([500/9, 100])
    inputs['bulan'] = '2'
    assert similar_articles(pd.DataFrame(rows), inputs).equals(found)
    inputs = dict.fromkeys(FEATURES, '0')
    rows[0].update(dict.fromkeys(FEATURES, None))
    assert similar_articles(pd.DataFrame(rows), inputs).iloc[0]['skor_kemiripan'] == 100


def test_bundle_uses_best_model_and_training_only(bundle):
    from sklearn.model_selection import train_test_split
    eligible = prepare_data(raw_data(), date(2026, 10, 5))
    train_idx, test_idx = train_test_split(eligible.index, test_size=0.3, random_state=43)
    actual = set(bundle['training_articles']['artikel'])
    assert actual == set(eligible.loc[train_idx, 'artikel'])
    assert not actual.intersection(eligible.loc[test_idx, 'artikel'])
    assert bundle['model_name'] == 'best_rf_mod'
    assert bundle['model'].get_params()['max_depth'] == 3
    assert bundle['metadata']['training_rows'] == 28
    assert bundle['metadata']['test_rows'] == 12
    assert all(pd.to_datetime(bundle['training_articles']['tanggal']) <= pd.Timestamp('2026-07-07'))
    pred = bundle['model'].predict(encode(bundle, bundle['training_articles']))
    assert pred.shape == (28, 4) and np.isfinite(pred).all()


class FakeBlob:
    def __init__(self, bucket, name, generation=None):
        self.bucket, self.name, self.generation = bucket, name, generation

    def upload_from_string(self, payload, content_type, if_generation_match):
        if self.bucket.fail_upload and self.name != MANIFEST:
            raise RuntimeError('Upload failed')
        current = self.bucket.data.get(self.name)
        assert if_generation_match == (current[1] if current else 0)
        self.generation = (current[1] if current else 0) + 1
        self.bucket.data[self.name] = (payload, self.generation)

    def download_as_bytes(self):
        payload, generation = self.bucket.data[self.name]
        assert generation == self.generation
        return payload


class FakeBucket:
    def __init__(self):
        self.data = {}
        self.fail_upload = False

    def bucket(self, name):
        return self

    def blob(self, name, generation=None):
        return FakeBlob(self, name, generation)

    def get_blob(self, name):
        return FakeBlob(self, name, self.data[name][1]) if name in self.data else None


def test_publish_download_and_failed_upload_keeps_pointer(bundle, tmp_path):
    path = tmp_path / 'bundle.joblib'
    joblib.dump(bundle, path)
    bucket = FakeBucket()
    manifest = publish_bundle(bucket, 'test', path, bundle['metadata'])
    restored = download_bundle(bucket, 'test', manifest)
    assert restored['metadata'] == bundle['metadata']
    previous = bucket.data[MANIFEST]
    bucket.fail_upload = True
    with pytest.raises(RuntimeError):
        publish_bundle(bucket, 'test', path, {**bundle['metadata'], 'version': 'new'})
    assert bucket.data[MANIFEST] == previous
    with pytest.raises(ValueError, match='Checksum'):
        download_bundle(bucket, 'test', {**manifest, 'sha256': 'bad'})
    with pytest.raises(ValueError, match='Versi'):
        download_bundle(bucket, 'test', {**manifest, 'library_versions': {'scikit-learn': '0.0'}})


def test_streamlit_form_predictions_and_similar_table(bundle, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv('MODEL_BUCKET', raising=False)
    root = Path(__file__).resolve().parents[1]
    for name in ['app.py', 'model_utils.py', 'model_store.py']:
        shutil.copy(root / name, tmp_path / name)
    joblib.dump(bundle, tmp_path / 'model_bundle.joblib')
    app = AppTest.from_file(str(tmp_path / 'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.selectbox) == 10
    app.button[0].click().run()
    assert any('Lengkapi' in x.value for x in app.warning)
    wanted = bundle['training_articles'].iloc[0]
    for selector, name in zip(app.selectbox, FEATURES):
        selector.select(str(wanted[name]))
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 5
    assert len(app.dataframe) == 1
    assert int(app.metric[-1].value) > 0
