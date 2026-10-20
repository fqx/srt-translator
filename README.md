# SRT Translator

This Python program translates subtitle files (SRT format) from one language to another using OpenAI's language model. It includes features such as movie information retrieval, context-aware translation, and progress tracking.

## Features

- Translates SRT subtitle files to a specified target language
- Retrieves movie information from TMDB to provide context for translation
- Splits subtitle content into manageable chunks for efficient processing
- Uses OpenAI's API for high-quality, context-aware translations
- Maintains subtitle formatting and timing
- Provides progress tracking and logging
- Supports batch processing of multiple SRT files

## Requirements

- Python 3.10+ (offline smoke tests verified with Python 3.11)
- OpenAI API key
- TMDB API key

## Installation

1. Clone this repository:

```
git clone https://github.com/fqx/srt-translator.git
cd srt-translator
```

2. Install the required packages:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

3. Create a `.env` file in the project root and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TMDB_API_KEY=your_tmdb_api_key
```

## Usage

Run the script with the following command:

python main.py -i <input_file_or_directory> -o <output_file> -l <target_language>

Arguments:

- `-i` or `--input`: Input SRT file or directory (default: current directory)
- `-o` or `--output`: Output SRT file (ignored if input is a directory)
- `-l` or `--language`: Target language (default: zh-CN for Simplified Chinese)

Example:

python main.py -i “Movie_Subtitle.srt” -o “Movie_Subtitle_Translated.srt” -l “fr”

## Notes

- The program uses the filename to identify the movie and retrieve its plot summary from TMDB. Ensure that your subtitle files are named accurately.
- The translation process may take some time depending on the length of the subtitle file and the API response times.
- Make sure you have sufficient credits in your OpenAI account, as the translation process consumes API tokens.

## Offline tests

Run `.venv/bin/python -m unittest discover -s tests -v` and
`.venv/bin/python main.py --help` after installation. The tests use placeholder
keys, disable dotenv loading and inherited proxies, and exercise subtitle
translation and TMDB requests through a loopback mock HTTP server. No API
account or paid service is needed. The server requires permission to bind a
loopback socket.

The proxy tests mock TLS handshakes to check urllib3 2.8.0's proxy-specific
certificate policy in both CONNECT and HTTPS forwarding modes. This is not a
live corporate-proxy interoperability test. Requests/TMDB uses CONNECT by
default; the aiohttp translation path does not use urllib3. If using a custom
urllib3 HTTPS forwarding proxy, configure proxy CAs and client certificates in
`proxy_ssl_context`, and proxy identity in `proxy_assert_hostname` or
`proxy_assert_fingerprint`. Destination credentials and identity overrides no
longer apply to the proxy handshake. See the [upstream release notes](https://github.com/urllib3/urllib3/releases/tag/2.8.0).

## License

[MIT License](LICENSE)

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/fqx/srt-translator/issues) if you want to contribute.
