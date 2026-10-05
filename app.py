import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import FEATURES, CHARACTERISTICS, LABELS, MONTHS, encode, month_options, similar_articles
from model_store import MANIFEST, download_bundle

st.set_page_config(page_title='Prediksi STR & Stok', layout='wide')


def setting(name, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except FileNotFoundError:
        return os.environ.get(name, default)


@st.cache_resource
def storage_client():
    from google.cloud import storage
    from google.oauth2 import service_account
    info = setting('gcp_service_account')
    if info:
        credentials = service_account.Credentials.from_service_account_info(dict(info))
        return storage.Client(project=info['project_id'], credentials=credentials)
    return storage.Client()


@st.cache_data(ttl=60, show_spinner=False)
def active_manifest(bucket):
    return json.loads(storage_client().bucket(bucket).blob(MANIFEST).download_as_text())


@st.cache_resource(max_entries=2, show_spinner='Memuat model terbaru...')
def cloud_bundle(bucket, manifest_json):
    return download_bundle(storage_client(), bucket, json.loads(manifest_json))


@st.cache_resource(max_entries=2)
def local_bundle(path, mtime_ns):
    return joblib.load(path)


def load_active():
    bucket = setting('MODEL_BUCKET')
    if bucket:
        try:
            manifest = active_manifest(bucket)
            bundle = cloud_bundle(bucket, json.dumps(manifest, sort_keys=True))
            st.session_state['last_good_bundle'] = bundle
            return bundle
        except Exception:
            logging.exception('Gagal mengambil model aktif dari Cloud Storage')
            st.warning('Model terbaru belum dapat dimuat. Menggunakan model terakhir yang tersedia jika ada.')
            if 'last_good_bundle' in st.session_state:
                return st.session_state['last_good_bundle']
    else:
        st.info('Mode model lokal. Pembaruan otomatis aktif setelah MODEL_BUCKET dan akses Cloud Storage dikonfigurasi.')
    path = Path(__file__).with_name('model_bundle.joblib')
    if path.exists():
        return local_bundle(str(path), path.stat().st_mtime_ns)
    st.error('Model belum tersedia. Jalankan training pertama dan periksa konfigurasi penyimpanan.')
    st.stop()


st.title('Prediksi STR & Stok')
st.write('Pilih karakteristik produk dan bulan launching untuk melihat prediksi serta artikel training yang serupa.')
bundle = load_active()
if bundle.get('model_name') != 'best_rf_mod':
    st.warning('Model lokal lama masih aktif. best_rf_mod dan daftar artikel serupa tersedia setelah training pertama berhasil.')
metadata = bundle.get('metadata', {})
if metadata:
    st.caption(f"Model: {bundle['model_name']} · Dilatih: {metadata['trained_at']} · {metadata['training_rows']} artikel training")

options = bundle['dropdown_options']
version = metadata.get('version', 'legacy')
if st.session_state.get('form_model_version') != version:
    for name in FEATURES:
        st.session_state.pop('input_' + name, None)
    st.session_state['form_model_version'] = version

with st.form('product'):
    inputs = {}
    boxes = st.columns(2)
    for i, name in enumerate(FEATURES):
        values = month_options(options[name]) if name == 'bulan' else options[name]
        with boxes[i % 2]:
            inputs[name] = st.selectbox(LABELS[name], values, index=None, placeholder='Pilih...',
                key='input_' + name,
                format_func=(lambda value: MONTHS[int(value)] if value.isdigit() and 1 <= int(value) <= 12 else value)
                            if name == 'bulan' else str)
    submitted = st.form_submit_button('Prediksi', type='primary')

if submitted:
    if any(value is None for value in inputs.values()):
        st.warning('Lengkapi semua pilihan terlebih dahulu.')
    else:
        raw = pd.DataFrame([inputs], columns=FEATURES)
        prediction = np.asarray(bundle['model'].predict(encode(bundle, raw)))[0]
        if len(prediction) != 4 or not np.isfinite(prediction).all():
            st.error('Model menghasilkan output tidak valid.')
            st.stop()
        result = dict(zip(bundle['target_columns'], prediction))
        st.subheader('Hasil prediksi')
        for box, target, label in zip(st.columns(3), ['str30', 'str60', 'str90'],
                                     ['STR 30 hari', 'STR 60 hari', 'STR 90 hari']):
            box.metric(label, f'{result[target]:.1%}')
        st.metric('Estimasi stok awal historis', f"{result['stock']:,.0f} pcs")
        st.caption('Estimasi stok mengikuti data historis; belum merupakan jumlah produksi atau restock optimal.')
        st.subheader('Artikel training yang mirip')
        articles = bundle.get('training_articles')
        if articles is None:
            st.info('Referensi artikel belum tersedia pada model lama. Jalankan training terbaru.')
        else:
            matches = similar_articles(articles, inputs)
            st.metric('Jumlah artikel mirip (>50%)', matches['artikel'].nunique())
            st.caption('Skor = jumlah karakteristik yang sama / 9 × 100%. Bulan tidak dihitung; 0 sama dengan 0 ikut dihitung. Urutan: launching terbaru.')
            if matches.empty:
                st.info('Belum ada artikel training dengan skor kemiripan di atas 50%.')
            else:
                shown = matches[['artikel', 'tanggal', 'skor_kemiripan', 'jumlah_sama'] + CHARACTERISTICS].copy()
                shown['tanggal'] = shown['tanggal'].dt.strftime('%Y-%m-%d')
                shown['skor_kemiripan'] = shown['skor_kemiripan'].map(lambda score: f'{score:.1f}%')
                shown = shown.rename(columns={**LABELS, 'artikel': 'Artikel', 'tanggal': 'Tanggal launching',
                    'skor_kemiripan': 'Kemiripan', 'jumlah_sama': 'Atribut sama (dari 9)'})
                st.dataframe(shown, hide_index=True, use_container_width=True)

st.caption('Pilihan berasal dari nilai unik df_reg yang lolos filter usia minimal 90 hari; null menjadi 0. Bulan diurutkan Januari–Desember sesuai nilai yang tersedia.')
