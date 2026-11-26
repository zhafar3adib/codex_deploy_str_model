"""List candidates, or activate a selected version without retraining."""
import argparse
import json
import os
from pathlib import Path

from model_store import activate_version, list_versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='klamby-469003')
    parser.add_argument('--bucket', default='klamby_sales')
    parser.add_argument('--version', default='')
    args = parser.parse_args()
    from google.cloud import storage
    client = storage.Client(project=args.project)
    selected = args.version.strip()
    if selected:
        result = activate_version(client, args.bucket, selected)
        summary = f"## Model diaktifkan\n\nVersi: `{result['version']}`\n\nRefresh Streamlit setelah cache manifest (60 detik) berakhir.\n"
    else:
        rows = list_versions(client, args.bucket)
        lines = ['## Versi model tersedia', '',
                 'Jalankan **Activate model** lagi dan isi **model_version** dengan ID pilihan.', '',
                 '| Versi | Status | MAE STR30 / STR60 / STR90 / stok |',
                 '|---|---|---|']
        for row in rows:
            metrics = row['metrics']
            scores = ' / '.join(str(round(metrics[t]['mae'], 4)) if t in metrics else '-' 
                                for t in ['str30', 'str60', 'str90', 'stock'])
            lines.append(f"| `{row['version']}` | {'Aktif' if row['active'] else 'Tersimpan'} | {scores} |")
        if not rows:
            lines.append('Belum ada model. Jalankan Train STR monthly terlebih dahulu.')
        lines += ['', 'Versi lama tanpa candidate.json tetap dapat dipilih; skor evaluasinya tersedia pada log training lama.',
                  '', 'Metrik lengkap untuk kandidat baru:', '', '```json', json.dumps(rows, indent=2), '```']
        summary = '\n'.join(lines) + '\n'
    print(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as stream:
            stream.write(summary)


if __name__ == '__main__':
    main()
