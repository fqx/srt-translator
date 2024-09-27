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

- Python 3.7+
- OpenAI API key
- TMDB API key

## Installation

1. Clone this repository:

```
git clone https://github.com/fqx/srt-translator.git
cd srt-translator
```

2. Install the required packages:

`pip install -r requirements.txt`

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

## License

[MIT License](LICENSE)

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/fqx/srt-translator/issues) if you want to contribute.
