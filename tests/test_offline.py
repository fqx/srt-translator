"""Offline dependency smoke tests: dummy configuration, loopback HTTP, mocked TLS."""
import asyncio
from contextlib import nullcontext
import importlib
import json
import socket
import ssl
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import certifi
import dotenv
import requests
import urllib3
from urllib3.connection import _WrappedAndVerifiedSocket

# Never read a developer's .env or use inherited API keys/proxy configuration.
with patch.dict('os.environ', {
    'OPENAI_API_KEY': 'offline-placeholder',
    'TMDB_API_KEY': 'offline-placeholder',
    'LOG_LEVEL': 'ERROR',
}, clear=True), patch.object(dotenv, 'load_dotenv'):
    app = importlib.import_module('main')


class Handler(BaseHTTPRequestHandler):
    calls = []
    rate_limit_once = False

    def log_message(self, *args):
        pass

    def reply(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Retry-After', '0')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).calls.append(self.path)
        if self.path.startswith('/search/movie'):
            self.reply({'results': [{'id': 1}]})
        elif self.path.startswith('/movie/1'):
            self.reply({'title': 'Example', 'overview': 'Offline movie plot.'})
        else:
            self.reply({}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).calls.append((self.path, payload, self.headers['Authorization']))
        if type(self).rate_limit_once:
            type(self).rate_limit_once = False
            self.reply({}, 429)
        else:
            self.reply({'choices': [{'message': {'content': json.dumps({
                'translation': '你好。<SEP>再见。', 'summary': 'Offline summary.'
            })}}]})


class OfflineTranslationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {}, clear=True)
        self.env.start()
        original_connect = socket.socket.connect

        def loopback_only(sock, address):
            if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
                raise AssertionError('External networking is forbidden in offline tests')
            return original_connect(sock, address)

        self.guard = patch.object(socket.socket, 'connect', loopback_only)
        self.guard.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.guard.stop)
        Handler.calls = []
        Handler.rate_limit_once = False
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_translation_pipeline_and_tmdb_transport(self):
        # Exercise real tmdbsimple -> requests -> urllib3 and real aiohttp HTTP.
        with requests.Session() as session, tempfile.TemporaryDirectory() as tmp:
            session.trust_env = False
            source = Path(tmp) / 'Example.2020.en.srt'
            output = Path(tmp) / 'translated.srt'
            source.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n'
                              '2\n00:00:03,000 --> 00:00:04,000\nGoodbye.\n')
            with patch.object(app, 'OPENAI_API_URL', self.url), \
                    patch.object(app.tmdb, 'REQUESTS_SESSION', session), \
                    patch('tmdbsimple.base.TMDB._get_complete_url',
                          lambda _, path: self.url + "/" + path.lstrip("/")):
                asyncio.run(app.main(str(source), str(output), 'zh-CN'))
            self.assertEqual(output.read_text(),
                             '1\n00:00:01,000 --> 00:00:02,000\n你好。\n\n'
                             '2\n00:00:03,000 --> 00:00:04,000\n再见。')
        posts = [call for call in Handler.calls if isinstance(call, tuple)]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0][0], '/v1/chat/completions')
        self.assertEqual(posts[0][2], 'Bearer offline-placeholder')
        self.assertIn('Offline movie plot.', posts[0][1]['messages'][1]['content'])
        self.assertTrue(any(isinstance(c, str) and c.startswith('/movie/1')
                            for c in Handler.calls))

    def test_rate_limit_retry(self):
        Handler.rate_limit_once = True

        async def run():
            async with app.aiohttp.ClientSession() as session:
                return await app.process_with_llm(session, 'Hello.<SEP>Goodbye.',
                                                  'Plot', 'zh-CN', 'Context')
        with patch.object(app, 'OPENAI_API_URL', self.url):
            self.assertEqual(asyncio.run(run()), ('你好。<SEP>再见。', 'Offline summary.'))
        self.assertEqual(len(Handler.calls), 2)

    def test_subtitle_chunking(self):
        chunks, times = app.split_subtitle(
            '1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n'
            '2\n00:00:03,000 --> 00:00:04,000\nGoodbye now', target_tokens=2)
        self.assertEqual(chunks, ['Hello world', 'Goodbye now'])
        self.assertEqual(len(times), 2)


class ProxyTLSTests(unittest.TestCase):
    def test_proxy_policy_does_not_inherit_destination_identity_or_credentials(self):
        # Check 2.8.0's security boundary without a real proxy or credentials.
        for forwarding in (False, True):
            with self.subTest(forwarding=forwarding):
                proxy_context = ssl.create_default_context()
                target_context = ssl.create_default_context()
                warning = (self.assertWarns(FutureWarning) if forwarding else nullcontext())
                with warning:
                    manager = urllib3.ProxyManager(
                        'https://proxy.invalid:443', proxy_ssl_context=proxy_context,
                        proxy_assert_hostname='proxy.invalid',
                        proxy_assert_fingerprint='11' * 32,
                        use_forwarding_for_https=forwarding,
                        ssl_context=target_context, assert_hostname='target.invalid',
                        assert_fingerprint='22' * 32,
                        cert_file='unused-destination-cert.pem',
                        key_file='unused-destination-key.pem')
                connection = manager.connection_from_url('https://target.invalid')._new_conn()
                if not forwarding:
                    connection.set_tunnel('target.invalid', 443, scheme='https')
                fake_socket = Mock()
                with patch.object(connection, '_new_conn', return_value=fake_socket), \
                        patch.object(connection, '_tunnel'), \
                        patch('urllib3.connection._ssl_wrap_socket_and_match_hostname',
                              return_value=_WrappedAndVerifiedSocket(fake_socket, True)) as wrap:
                    connection.connect()
                proxy_calls = [c.kwargs for c in wrap.call_args_list
                               if c.kwargs.get('server_hostname') == 'proxy.invalid']
                self.assertEqual(len(proxy_calls), 1)
                policy = proxy_calls[0]
                self.assertIs(policy['ssl_context'], proxy_context)
                self.assertEqual(policy['cert_reqs'], ssl.CERT_REQUIRED)
                self.assertEqual(policy['assert_hostname'], 'proxy.invalid')
                self.assertEqual(policy['assert_fingerprint'], '11' * 32)
                self.assertIsNone(policy['cert_file'])
                self.assertIsNone(policy['key_file'])
                self.assertEqual(proxy_context.verify_mode, ssl.CERT_REQUIRED)
                manager.clear()

    def test_requests_uses_tunneling_proxy_by_default(self):
        adapter = requests.adapters.HTTPAdapter()
        self.addCleanup(adapter.close)
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('No network')):
            manager = adapter.proxy_manager_for('https://proxy.invalid:443')
            self.assertFalse(manager.proxy_config.use_forwarding_for_https)
            self.assertIsNone(manager.proxy_config.ssl_context)


class CertificateBundleTests(unittest.TestCase):
    def test_bundle_removes_compromised_root_and_retains_unaffected_roots(self):
        context = ssl.create_default_context(cafile=certifi.where())
        names = {value for cert in context.get_ca_certs()
                 for rdn in cert['subject'] for key, value in rdn
                 if key == 'commonName'}
        self.assertNotIn('emSign ECC Root CA - G3', names)
        self.assertTrue({'emSign Root CA - G1', 'emSign Root CA - C1',
                         'emSign ECC Root CA - C3'}.issubset(names))
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_requests_default_ca_path_uses_installed_certifi(self):
        bundle = Path(certifi.where()).resolve()
        self.assertEqual(Path(requests.certs.where()).resolve(), bundle)
        self.assertEqual(Path(requests.adapters.DEFAULT_CA_BUNDLE_PATH).resolve(), bundle)
        adapter = requests.adapters.HTTPAdapter()
        self.addCleanup(adapter.close)
        connection = Mock()
        adapter.cert_verify(connection, 'https://offline.invalid', True, None)
        self.assertEqual(Path(connection.ca_certs).resolve(), bundle)
        self.assertEqual(connection.cert_reqs, 'CERT_REQUIRED')


if __name__ == '__main__':
    unittest.main()
