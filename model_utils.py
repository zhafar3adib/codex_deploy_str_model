"""Shared input contract and article similarity for training and inference."""
import numpy as np
import pandas as pd

FEATURES = ['sub_kategori', 'cutting', 'design_density', 'design_border',
            'siluet', 'sleeve', 'collar', 'length_hem', 'color_theme', 'bulan']
CHARACTERISTICS = FEATURES[:-1]
TARGETS = ['str30', 'str60', 'str90', 'stock']
MONTHS = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
          'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
LABELS = dict(zip(FEATURES, ['Kategori produk', 'Cutting / potongan', 'Kepadatan motif',
    'Border motif', 'Siluet', 'Lengan', 'Kerah', 'Panjang hem', 'Tema warna', 'Bulan launching']))


def normalized_inputs(frame):
    # Object first preserves missing values as the exact category "0".
    return frame[FEATURES].astype(object).where(frame[FEATURES].notna(), 0).astype(str)


def month_options(values):
    return sorted(set(map(str, values)), key=lambda v: (0, int(v))
                  if v.isdigit() and 1 <= int(v) <= 12 else (1, v))


def dropdowns(frame):
    normalized = normalized_inputs(frame)
    options = {c: normalized[c].unique().tolist() for c in FEATURES}
    options['bulan'] = month_options(options['bulan'])
    return options


def prepare_data(raw, as_of):
    data = raw.copy()
    data['tanggal'] = pd.to_datetime(data['tanggal'], errors='coerce').dt.normalize()
    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=90)
    data = data.loc[data['tanggal'].notna() & (data['tanggal'] <= cutoff)].copy()
    data['stock'] = pd.to_numeric(data['stock'], errors='coerce')
    data = data.loc[data['artikel'].notna() & np.isfinite(data['stock']) & (data['stock'] > 0)].copy()
    if data['artikel'].duplicated().any():
        raise ValueError('Artikel duplikat pada data training; periksa mapping kategori/SKU di sumber data.')
    data = data.reset_index(drop=True)
    data['bulan'] = data['tanggal'].dt.month.astype(str)
    data[FEATURES] = normalized_inputs(data)
    for days in (30, 60, 90):
        quantity = pd.to_numeric(data[f'quantity{days}'], errors='raise').fillna(0)
        if not np.isfinite(quantity).all() or (quantity < 0).any():
            raise ValueError('Quantity training harus finite dan non-negatif.')
        data[f'str{days}'] = (quantity / data['stock']).clip(upper=1)
    return data


def encode(bundle, frame):
    values = bundle['encoder'].transform(normalized_inputs(frame))
    if hasattr(values, 'toarray'):
        values = values.toarray()
    return pd.DataFrame(values, columns=bundle['encoder'].get_feature_names_out(FEATURES),
                        index=frame.index)[bundle['feature_columns']]


def similar_articles(training_articles, inputs):
    """Exact equal-weight matches; exclude month, threshold strictly >50%."""
    result = training_articles.copy()
    attributes = normalized_inputs(result)[CHARACTERISTICS]
    wanted = pd.Series({c: str(inputs[c]) for c in CHARACTERISTICS})
    result['jumlah_sama'] = attributes.eq(wanted, axis='columns').sum(axis=1)
    result['skor_kemiripan'] = result['jumlah_sama'] / len(CHARACTERISTICS) * 100
    result = result.loc[result['skor_kemiripan'] > 50].copy()
    result['tanggal'] = pd.to_datetime(result['tanggal'])
    return result.sort_values(['tanggal', 'artikel'], ascending=[False, True]).reset_index(drop=True)
