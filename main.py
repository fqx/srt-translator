import os, glob
import re
import asyncio
import aiohttp
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import random
import time
import argparse
from langcodes import Language
import json
import tmdbsimple as tmdb
from fuzzywuzzy import fuzz
import chardet
import guessit

# load env
load_dotenv()

# get api keys
API_KEY = os.getenv("OPENAI_API_KEY")
tmdb.API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# set logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# Semaphore for controlling concurrency.
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def read_file_with_auto_encoding(file_path):
    # Try to detect the file's encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()

    detected = chardet.detect(raw_data)
    encoding = detected['encoding']

    # List of encodings to try if detection fails or gives an error
    fallback_encodings = ['utf-8', 'GBK', 'iso-8859-1', 'windows-1252', 'ascii']

    # Try the detected encoding first, then fall back to others if it fails
    encodings_to_try = [encoding] + [e for e in fallback_encodings if e != encoding]

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    # If all encodings fail, raise an exception
    raise ValueError(f"Unable to decode the file {file_path} with any of the attempted encodings.")


def get_movie_info(filename):
    """
    From the subtitle filename, identify the movie/TV show and get its plot summary from TMDB
    """
    base_name = os.path.basename(filename)
    movie_name = os.path.splitext(base_name)[0]
    logging.info(f"Identifying movie: {movie_name}")
    guess = guessit.guessit(movie_name)
    title = guess.get('title')
    year = guess.get('year')

    # Determine if it's a movie or TV show
    type = guess.get('type', 'movie')  # Default to movie if type is not specified

    # Search for the movie
    search = tmdb.Search()
    response = search.movie(query=movie_name)

    if type == 'episode':
        # For TV shows, we need to search for the series first
        if year:
            response = search.tv(query=title, first_air_date_year=year)
        else:
            response = search.tv(query=title)

        if search.results:
            show = search.results[0]
            show_details = tmdb.TV(show['id']).info()

            # Try to get specific episode info if available
            season_number = guess.get('season')
            episode_number = guess.get('episode')

            if season_number and episode_number:
                episode = tmdb.TV_Episodes(show['id'], season_number, episode_number).info()
                return f"Series: {show_details['name']}\nOverview: {show_details['overview']}\n\nEpisode: {episode.get('name', 'Unknown')}\nEpisode Plot: {episode.get('overview', 'No specific episode plot available.')}"
            else:
                return f"Series: {show_details['name']}\nOverview: {show_details['overview']}"
        else:
            return "No TV show information found."
    else:
        # For movies
        if year:
            response = search.movie(query=title, year=year)
        else:
            response = search.movie(query=title)

        if search.results:
            movie = search.results[0]
            movie_details = tmdb.Movies(movie['id']).info()
            return f"Movie: {movie_details['title']}\nOverview: {movie_details['overview']}"
        else:
            return f"No plot summary available for {movie_name}"


def split_subtitle(subtitle_content, target_tokens=200):
    """
    Split the subtitle file into chunks of approximately 200-300 tokens,
    remove time codes and format characters, but maintain line breaks.
    """
    subtitle_parts = re.split(r'\n\n(?=\d+\n)', subtitle_content.strip())
    cleaned_parts = []
    time_codes = []
    current_chunk = []
    current_tokens = 0

    for part in subtitle_parts:
        lines = part.split('\n')
        if len(lines) >= 3:
            time_codes.append('\n'.join(lines[:2]))
            text = ' '.join(lines[2:])
            # text = text.replace('|', '\\|')  # Escape any existing '|' characters
            text_tokens = len(text.split())

            if current_tokens + text_tokens > target_tokens and current_chunk:
                cleaned_parts.append('<SEP>'.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(text)
            current_tokens += text_tokens

    if current_chunk:
        cleaned_parts.append('<SEP>'.join(current_chunk))

    return cleaned_parts, time_codes


async def process_with_llm(session, text, story_background, language, context, max_retries=3):
    """
    Use LLM (OpenAI API) for text translation and context generation, including retry logic.
    """
    lang = Language.get(language).display_name()

    prompt = f"""
    Story background: {story_background}
    Current context: {context}
    Translate the following text into {lang}, maintaining the tone and style appropriate for the story. 
    The '<SEP>' represents a seperator in the subtitle. Please PRESERVE these seperator and DO NOT add new seperator in your translation.
    Then, based on the translation, provide a brief update of the context:

    Text to translate:
    {text}

    Translation:

    Updated context (briefly summarize the key points from this segment, max 50 words):
    """

    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                        f"{OPENAI_API_URL}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": "You are a professional translator."},
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "max_tokens": 2000,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "srt_translate",
                                    "strict": True,
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "translation": {
                                                "type": "string",
                                                "description": "The translated text"
                                            },

                                            "summary": {
                                                "type": "string",
                                                "description": "A brief summary of the previous and current context"
                                            },

                                        },
                                        "required": ["translation", "summary"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                        }
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logging.warning(f"Rate limit reached. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue

                    result = await response.json()
                    content = result['choices'][0]['message']['content']

                    # Parse the JSON response
                    try:
                        parsed_content = json.loads(content)
                        translation = parsed_content['translation']
                        new_context_summary = parsed_content['summary']

                        # Log the translation for verification
                        logging.info(f"Translated: {translation}")
                        logging.info(f"New context: {new_context_summary}")

                        return translation, new_context_summary
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse JSON response: {e}")
                        logging.error(f"Raw content: {content}")
                        return None, context
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Error in LLM processing: {e}. Retrying in {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"Failed to process after {max_retries} attempts: {e}")
                return text, context

    return text, context  # If all retries fail, return the original text and context.


async def validation(translated_chunk, original_chunk, movie_info, language, context, max_retries=3):
    """
    check the validation of translated_chunk
    """
    attempt=1
    translated_chunk_list = re.split(r'<SEP>', translated_chunk)
    original_chunk_list = re.split(r'<SEP>', original_chunk)

    while len(translated_chunk_list) != len(original_chunk_list):
        if attempt > max_retries:
        # align translated chunk to original chunk
            logging.warning("Align translated chunk arbitrarily.")
            if len(translated_chunk_list) > len(original_chunk_list):
                # Cut off A if it's longer than B
                translated_chunk_list = translated_chunk_list[:len(original_chunk_list)]
            elif len(translated_chunk_list) < len(original_chunk_list):
                # Add empty strings to A if it's shorter than B
                translated_chunk_list.extend([''] * (len(original_chunk_list) - len(translated_chunk_list)))

            return '<SEP>'.join(translated_chunk_list)

        logging.warning(f"Validation failed. {attempt} attempts to get a new translation.")
        async with aiohttp.ClientSession() as session:
            translated_chunk, context = await translate_subtitle_part(session, original_chunk, movie_info, language, context)
            try:
                translated_chunk_list = re.split(r'<SEP>', translated_chunk)
            except TypeError:
                translated_chunk_list = original_chunk_list.append('')

        attempt += 1

    else:
        logging.info("Validation passed.")
        return translated_chunk


async def reassemble_subtitle(translated_parts, time_codes, subtitle_parts, movie_info, language, context):
    """
    Reassemble the translated text with time codes and format characters
    """
    reassembled = []
    time_code_index = 0

    tasks = [validation(translated_chunk, original_chunk, movie_info, language, context) for (translated_chunk, original_chunk) in zip(translated_parts, subtitle_parts)]
    translated_parts = await asyncio.gather(*tasks)

    for idx, (translated_chunk, original_chunk) in enumerate(zip(translated_parts, subtitle_parts)):

        # Split the chunk into individual subtitle texts, respecting escaped '<SEP>>'

        if translated_chunk is not None:
            subtitle_texts = re.split(r'<SEP>', translated_chunk)
        else:
            subtitle_texts = re.split(r'<SEP>', original_chunk)

        # Reassemble each subtitle text with its time code
        for subtitle_text in subtitle_texts:
            if time_code_index < len(time_codes):
                time_code = time_codes[time_code_index]
                reassembled.append(f"{time_code}\n{subtitle_text.strip()}")
                time_code_index += 1
            else:
                # Handle the case where we run out of time codes
                logging.warning(f"Ran out of time codes at index {time_code_index}")
                break

    return '\n\n'.join(reassembled)


async def translate_subtitle_part(session, part, story_background, language, context):
    """
    Translate individual subtitle parts.
    """
    translation, new_context = await process_with_llm(session, part, story_background, language, context)
    return translation, new_context


async def main(input_path, output_file, language):
    """
    Main function to coordinate the entire translation process
    """
    if os.path.isdir(input_path):
        srt_files = glob.glob(os.path.join(input_path, '*.srt'))
    elif os.path.isfile(input_path):
        srt_files = [input_path]
    else:
        srt_files = glob.glob('*.srt')

    for input_file in tqdm(srt_files):
        if output_file is None or os.path.isdir(input_path):
            input_base, input_ext = os.path.splitext(input_file)
            # Remove original language code if present
            input_base = input_base.split('.')
            if Language.get(input_base[-1]).is_valid():
                input_base = '.'.join(input_base[:-1])
            else:
                input_base = '.'.join(input_base)
            # Get the 3-letter language code
            lang_code = Language.get(language).to_tag()

            # Create the new output filename
            output_file = f"{input_base}.{lang_code}{input_ext}"

        movie_info = get_movie_info(input_file)
        context = movie_info

        subtitle_content = read_file_with_auto_encoding(input_file)

        subtitle_parts, time_codes = split_subtitle(subtitle_content)

        async with aiohttp.ClientSession() as session:
            translated_parts = []
            for part in subtitle_parts:
                translation, context = await translate_subtitle_part(session, part, movie_info, language, context)
                translated_parts.append(translation)

        final_subtitle = await reassemble_subtitle(translated_parts, time_codes, subtitle_parts, movie_info, language, context)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_subtitle)

        logging.info(f"Translation completed. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SRT files.")
    parser.add_argument("-i", "--input", default=".", help="Input SRT file or directory")
    parser.add_argument("-o", "--output", default=None, help="Output SRT file (ignored if input is a directory)")
    parser.add_argument("-l", "--language", default="zh-CN", help="Target language")

    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.language))
