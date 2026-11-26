"""Versioned private GCS artifacts; activate only a fully uploaded bundle."""
import hashlib
from importlib.metadata import version
import io
import json
import re

import joblib

MANIFEST = 'str-model/latest.json'


def publish_bundle(client, bucket_name, path, metadata):
    bucket = client.bucket(bucket_name)
    validate_version(metadata['version'])
    payload = path.read_bytes()
    object_name = f"str-model/versions/{metadata['version']}/model_bundle.joblib"
    blob = bucket.blob(object_name)
    blob.upload_from_string(payload, content_type='application/octet-stream', if_generation_match=0)
    manifest = {
        'version': metadata['version'], 'object': object_name,
        'generation': int(blob.generation), 'sha256': hashlib.sha256(payload).hexdigest(),
        'library_versions': metadata['library_versions'], 'trained_at': metadata['trained_at'],
        'metrics': metadata.get('metrics', {}),
    }
    # Training stores an immutable candidate; it never changes the active pointer.
    bucket.blob(f"str-model/versions/{metadata['version']}/candidate.json").upload_from_string(
        json.dumps(manifest, allow_nan=False), content_type='application/json', if_generation_match=0)
    return manifest


def check_versions(manifest):
    for package, expected in manifest['library_versions'].items():
        if version(package) != expected:
            raise ValueError(f'Versi {package} berbeda dari training; sesuaikan requirements.txt aplikasi.')


def download_bundle(client, bucket_name, manifest):
    check_versions(manifest)
    name = manifest['object']
    if not name.startswith('str-model/versions/'):
        raise ValueError('Lokasi model tidak valid.')
    blob = client.bucket(bucket_name).blob(name, generation=int(manifest['generation']))
    payload = blob.download_as_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('Checksum model tidak sesuai; versi baru tidak digunakan.')
    bundle = joblib.load(io.BytesIO(payload))
    if bundle.get('schema_version') != 2 or bundle.get('model_name') != 'best_rf_mod':
        raise ValueError('Bundle bukan best_rf_mod dengan format terbaru.')
    if bundle['metadata']['version'] != manifest['version']:
        raise ValueError('Versi bundle tidak sesuai manifest.')
    return bundle


def validate_version(model_version):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', model_version):
        raise ValueError('ID versi tidak valid. Salin nama folder versi, bukan URL atau path.')


def candidate_manifest(client, bucket_name, model_version):
    """Support both new candidates and existing pre-manual-activation bundles."""
    validate_version(model_version)
    bucket = client.bucket(bucket_name)
    prefix = f'str-model/versions/{model_version}/'
    record = bucket.get_blob(prefix + 'candidate.json')
    if record is not None:
        manifest = json.loads(record.download_as_bytes())
        if manifest['version'] != model_version or manifest['object'] != prefix + 'model_bundle.joblib':
            raise ValueError('Metadata kandidat tidak sesuai versi yang dipilih.')
        return manifest
    blob = bucket.get_blob(prefix + 'model_bundle.joblib')
    if blob is None:
        raise ValueError('Versi model tidak ditemukan di bucket.')
    # Legacy artifacts have no candidate.json. Read the pinned generation from this trusted bucket.
    payload = bucket.blob(blob.name, generation=int(blob.generation)).download_as_bytes()
    bundle = joblib.load(io.BytesIO(payload))
    if bundle.get('schema_version') != 2 or bundle.get('model_name') != 'best_rf_mod':
        raise ValueError('Model lama tidak memiliki format bundle yang didukung.')
    metadata = bundle['metadata']
    if metadata['version'] != model_version:
        raise ValueError('Versi dalam bundle tidak sesuai nama folder.')
    return {
        'version': model_version, 'object': blob.name, 'generation': int(blob.generation),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'library_versions': metadata['library_versions'], 'trained_at': metadata['trained_at'],
        'metrics': metadata.get('metrics', {}),
    }


def activate_version(client, bucket_name, model_version):
    """Validate first, then atomically switch the active pointer; no training."""
    validate_version(model_version)
    bucket = client.bucket(bucket_name)
    previous = bucket.get_blob(MANIFEST)
    expected_generation = int(previous.generation) if previous else 0
    manifest = candidate_manifest(client, bucket_name, model_version)
    bundle = download_bundle(client, bucket_name, manifest)
    import numpy as np
    from model_utils import encode, TARGETS
    if bundle.get('target_columns') != TARGETS:
        raise ValueError('Target model tidak sesuai aplikasi.')
    sample = bundle['training_articles'].iloc[:1]
    if sample.empty:
        raise ValueError('Referensi training kosong.')
    prediction = np.asarray(bundle['model'].predict(encode(bundle, sample)))
    if prediction.shape != (1, 4) or not np.isfinite(prediction).all():
        raise ValueError('Prediksi uji model tidak valid; model aktif tidak diubah.')
    bucket.blob(MANIFEST).upload_from_string(
        json.dumps(manifest, allow_nan=False), content_type='application/json',
        if_generation_match=expected_generation)
    return manifest


def list_versions(client, bucket_name):
    """Show available versions without loading model binaries."""
    bucket = client.bucket(bucket_name)
    active_blob = bucket.get_blob(MANIFEST)
    active = json.loads(active_blob.download_as_bytes())['version'] if active_blob else None
    rows = []
    prefix = 'str-model/versions/'
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        relative = blob.name[len(prefix):].split('/')
        if len(relative) != 2 or relative[1] != 'model_bundle.joblib':
            continue
        model_version = relative[0]
        validate_version(model_version)
        record = bucket.get_blob(f'{prefix}{model_version}/candidate.json')
        metadata = json.loads(record.download_as_bytes()) if record else {}
        rows.append({'version': model_version, 'active': model_version == active,
                     'trained_at': metadata.get('trained_at'), 'metrics': metadata.get('metrics', {})})
    return sorted(rows, key=lambda row: row['version'], reverse=True)
