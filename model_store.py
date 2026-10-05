"""Versioned private GCS artifacts; activate only a fully uploaded bundle."""
import hashlib
from importlib.metadata import version
import io
import json

import joblib

MANIFEST = 'str-model/latest.json'


def publish_bundle(client, bucket_name, path, metadata):
    bucket = client.bucket(bucket_name)
    pointer = bucket.blob(MANIFEST)
    previous = bucket.get_blob(MANIFEST)
    expected_generation = int(previous.generation) if previous else 0
    payload = path.read_bytes()
    object_name = f"str-model/versions/{metadata['version']}/model_bundle.joblib"
    blob = bucket.blob(object_name)
    blob.upload_from_string(payload, content_type='application/octet-stream', if_generation_match=0)
    manifest = {
        'version': metadata['version'], 'object': object_name,
        'generation': int(blob.generation), 'sha256': hashlib.sha256(payload).hexdigest(),
        'library_versions': metadata['library_versions'], 'trained_at': metadata['trained_at'],
    }
    # A failed upload or concurrent publication cannot corrupt/overwrite the active version.
    pointer.upload_from_string(json.dumps(manifest), content_type='application/json',
                               if_generation_match=expected_generation)
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
