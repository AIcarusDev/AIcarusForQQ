from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from llm.media.image_store import (
    register_image, read_image, reserve_ref, register_entry,
    update_description, append_examination, description_claim,
)
from llm.media.media_identity import MediaRefConflict
from llm.media.media_cache import clear_recent_media_cache
from workspace.media import register_workspace_bytes


def png(color='red'):
    stream = io.BytesIO()
    Image.new('RGB', (12, 9), color).save(stream, 'PNG')
    return stream.getvalue()


def test_sources_aliases_and_restart_share_original():
    raw = png()
    old = reserve_ref()
    first = register_image(raw, 'chat', old)
    second_ref = reserve_ref()
    second = register_image(raw, 'browser', second_ref)
    third = asyncio.run(register_workspace_bytes(raw))
    assert first['image_ref'] == second['image_ref'] == third
    assert second['reused']
    clear_recent_media_cache()
    assert read_image(second_ref)['data'] == raw
    assert read_image(second_ref)['sha256'] == hashlib.sha256(raw).hexdigest()
    assert register_image(png('blue'), 'chat')['image_ref'] != first['image_ref']


def test_parallel_registration_only_one_original():
    raw = png()
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda _: register_image(raw, 'chat'), range(16)))
    assert len({r['image_ref'] for r in records}) == 1
    assert sum(not r['reused'] for r in records) == 1
    from llm.media.media_storage import MEDIA_ROOT
    assert len(list(MEDIA_ROOT.rglob('*.png'))) == 1


def test_immutable_binding_and_corruption():
    first = register_image(png(), 'chat')
    with pytest.raises(MediaRefConflict):
        register_image(png('blue'), 'chat', first['image_ref'])
    Path(first['locator']).write_bytes(png('blue'))
    assert read_image(first['image_ref'])['unavailable_status'] == 'image_changed'
    with pytest.raises(MediaRefConflict):
        register_image(png(), 'chat')


def test_duplicate_occurrences_preserve_order_and_share_information():
    raw = png()
    refs = [reserve_ref(), reserve_ref()]
    entry = {'content_segments': [{'type': 'image', 'image_ref': r} for r in refs],
             'images': {r: {'base64': base64.b64encode(raw).decode()} for r in refs}}
    register_entry(entry)
    canonical = entry['content_segments'][0]['image_ref']
    assert [s['image_ref'] for s in entry['content_segments']] == [canonical, canonical]
    assert len(entry['images']) == 1
    update_description(canonical, 'fixture description')
    append_examination(refs[1], 'fixture focus', 'fixture result')
    assert read_image(refs[0])['description'] == 'fixture description'
    assert read_image(refs[1])['examinations'][0]['result'] == 'fixture result'


def test_collection_removal_keeps_original():
    from llm.media.sticker_collection import save_sticker, delete_sticker
    ref = register_image(png(), 'workspace')['image_ref']
    assert save_sticker(png(), 'image/png', 'useful', image_ref=ref) == (ref, False)
    assert delete_sticker(ref) == ref
    assert read_image(ref)['data'] == png()


def test_description_claim_shared_by_concurrent_consumers():
    ref = register_image(png(), 'chat')['image_ref']
    def describe(_):
        with description_claim(ref) as acquired:
            if acquired:
                update_description(ref, 'one description')
            return acquired
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(describe, range(4))) == 1


def test_cross_process_content_deduplication(tmp_path):
    import subprocess
    import sys
    import database
    from llm.media import media_storage, sticker_collection
    source = tmp_path / 'input.png'
    source.write_bytes(png())
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import database
from llm.media import media_storage, sticker_collection
from llm.media.image_store import register_image
database.DB_PATH=sys.argv[2]
media_storage.MEDIA_ROOT=Path(sys.argv[3])
sticker_collection._INDEX_PATH=Path(sys.argv[4])
sticker_collection._IMAGES_DIR=Path(sys.argv[4]).parent/'images'
print(register_image(Path(sys.argv[5]).read_bytes(),'workspace')['image_ref'])
'''
    args = [sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parents[1]/'src'),
            database.DB_PATH, str(media_storage.MEDIA_ROOT), str(sticker_collection._INDEX_PATH), str(source)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        refs = list(executor.map(lambda _: subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip(), range(3)))
    assert len(set(refs)) == 1
    assert read_image(refs[0])['data'] == png()


def test_browser_confirmation_cannot_be_bypassed_with_alias_or_sticker(tmp_path):
    from browser.image_resources import BrowserImageArtifactStore, BrowserImageResourceRegistry
    from platforms.qq.tools.qq_social.send_message import send_message
    from types import SimpleNamespace
    raw = png()
    first = register_image(raw, 'workspace')
    resource = BrowserImageResourceRegistry().register(
        source_url='https://example.invalid/image.png', page_url='https://example.invalid',
        identity='test', alt='test', rect={'x':0,'y':0,'width':12,'height':9}, natural_size=(12,9))
    artifact = BrowserImageArtifactStore(tmp_path/'browser').persist(raw, resource=resource, strategy='response_body', declared_mime='image/png')
    assert artifact.image_ref == first['image_ref']
    assert artifact.confirmation_reasons
    alias = reserve_ref()
    register_image(raw, 'chat', alias)
    # Exercise the same guard used by the send handler, without sending anything.
    for command in ('image', 'sticker'):
        messages = [{'segments':[{'command':command,'image_ref':alias}]}]
        assert send_message._unconfirmed_high_risk_image_error(messages, SimpleNamespace(), {"browser_control": {"image_send_confirmation": "high_risk"}})
