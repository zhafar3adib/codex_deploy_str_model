from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Prediksi STR & Stok', layout='centered')

@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).with_name('model_bundle.joblib'))

st.title('Prediksi STR & Stok')
st.write('Pilih atribut produk dan bulan peluncuran, lalu klik Prediksi.')
if not Path(__file__).with_name('model_bundle.joblib').exists():
    st.error('File model belum tersedia. Jalankan export_colab.py di notebook, lalu tambahkan model_bundle.joblib ke folder aplikasi.')
    st.stop()

bundle = load_bundle()
encoder = bundle['encoder']
cat_cols = bundle['categorical_columns']
if 'dropdown_options' not in bundle:
    st.error('Model bundle masih versi lama. Jalankan export_colab.py terbaru di Colab dan ganti model_bundle.joblib.')
    st.stop()
dropdown_options = bundle['dropdown_options']
labels = {
    'sub_kategori': 'Kategori produk', 'cutting': 'Cutting / potongan',
    'design_density': 'Kepadatan motif', 'design_border': 'Border motif',
    'siluet': 'Siluet', 'sleeve': 'Lengan', 'collar': 'Kerah',
    'length_hem': 'Panjang hem', 'color_theme': 'Tema warna',
    'bulan': 'Bulan peluncuran',
}
months = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
          'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

def display_value(value, column):
    if column == 'bulan' and value.isdigit() and 1 <= int(value) <= 12:
        return months[int(value)]
    return value

with st.form('product'):
    inputs = {}
    columns = st.columns(2)
    for i, name in enumerate(cat_cols):
        options = dropdown_options[name]
        with columns[i % 2]:
            inputs[name] = st.selectbox(
                labels.get(name, name), options, index=None,
                placeholder='Pilih...', key=name,
                format_func=lambda value, column=name: display_value(value, column),
            )
    submitted = st.form_submit_button('Prediksi', type='primary')

if submitted:
    if any(value is None for value in inputs.values()):
        st.warning('Lengkapi semua pilihan terlebih dahulu.')
    else:
        raw = pd.DataFrame([inputs], columns=cat_cols).astype(str)
        values = encoder.transform(raw)
        if hasattr(values, 'toarray'):
            values = values.toarray()
        encoded = pd.DataFrame(values, columns=encoder.get_feature_names_out(cat_cols))
        prediction = np.asarray(bundle['model'].predict(encoded[bundle['feature_columns']]))[0]
        if len(prediction) != 4 or not np.isfinite(prediction).all():
            st.error('Model menghasilkan output tidak valid. Periksa model di Colab.')
            st.stop()
        result = dict(zip(bundle['target_columns'], prediction))
        st.subheader('Hasil prediksi')
        for box, target, label in zip(st.columns(3), ['str30', 'str60', 'str90'],
                                      ['STR 30 hari', 'STR 60 hari', 'STR 90 hari']):
            box.metric(label, f'{result[target]:.1%}')
        st.metric('Estimasi stok awal historis', f"{result['stock']:,.0f} pcs")
        st.caption('Estimasi stok mengikuti pola stok awal data pelatihan; belum merupakan rekomendasi jumlah produksi atau restock optimal.')

st.caption('Pilihan berasal dari nilai unik setiap kolom input df_reg setelah null diganti 0. Kombinasi atribut baru tetap perlu ditinjau. Definisi periode STR mengikuti notebook pelatihan.')
