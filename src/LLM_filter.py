import json
import requests
import logging
from typing import Dict, List
import yaml
import os
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, 
                 input_file: str,
                 output_file: str,
                 master_prompt: str = None,
                 chunk_size: int = 5000,
                 continue_processing: bool = False):  # Added continue flag
        self.input_file = input_file
        self.output_file = output_file
        self.master_prompt = master_prompt
        self.chunk_size = chunk_size
        self.continue_processing = continue_processing
        
        # Track progress in a separate file
        self.progress_file = f"{output_file}.progress"
        self.processed_lines = self._load_progress()

    def _load_progress(self) -> int:
        """Load the number of previously processed lines"""
        if not self.continue_processing:
            return 0
        try:
            with open(self.progress_file, 'r') as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return 0

    def _save_progress(self, line_number: int):
        """Save the current progress"""
        with open(self.progress_file, 'w') as f:
            f.write(str(line_number))

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        # Split on sentence boundaries to avoid cutting mid-sentence
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '  # Restore the period
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def query_llm(self, text: str) -> str:
        """Query the Ollama API using the working curl format"""
        prompt = f"{self.master_prompt}\n\nInput text:\n{text}"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "phi4:latest",
                    # "model": "llama3.1:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.05
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return None

    def process_entry(self, entry: Dict) -> Dict:
        """Process a single JSON entry, handling long texts"""
        text = entry['text']
        
        # If text is short enough, process directly
        if len(text) <= self.chunk_size:
            processed_text = self.query_llm(text)
            if processed_text and processed_text != "NOT_RELEVANT":
                entry['text'] = processed_text
                return entry
            return None
        
        # For long texts, split into chunks and process each
        chunks = self.split_text(text)
        processed_chunks = []
        
        logger.info(f"Processing long text in {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            processed_chunk = self.query_llm(chunk)
            if not processed_chunk or processed_chunk == "NOT_RELEVANT":
                logger.warning(f"Chunk {i+1}/{len(chunks)} was marked as not relevant or failed processing")
                continue
            processed_chunks.append(processed_chunk)
        
        # If no chunks were successfully processed, return None
        if not processed_chunks:
            return None
        
        # Combine processed chunks
        entry['text'] = ' '.join(processed_chunks)
        return entry

    def process_entry_wrapper(self, args):
        """Wrapper for process_entry to work with multiprocessing"""
        line_number, line = args
        try:
            entry = json.loads(line.strip())
            processed_entry = self.process_entry(entry)
            return line_number, processed_entry
        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            return line_number, None

    def process_file(self):
        """Process entries of JSONL file in parallel"""
        processed_count = 0
        skipped = 0
        
        # Open file in append mode if continuing
        mode = 'a' if self.continue_processing else 'w'
        
        # Read all lines that need processing
        with open(self.input_file, 'r', encoding='utf-8') as infile:
            # Skip already processed lines if continuing
            for _ in range(self.processed_lines):
                next(infile)
                skipped += 1
            
            # Create list of (line_number, line) tuples to process
            lines_to_process = []
            for line_number, line in enumerate(infile, start=skipped):
                # if processed_count >= 100:  # Original limit
                #     break
                lines_to_process.append((line_number, line))
                processed_count += 1

        # Process entries in parallel
        num_workers = 3  # Adjust based on your GPU memory
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_entry_wrapper, item) 
                      for item in lines_to_process]
            
            with open(self.output_file, mode, encoding='utf-8') as outfile:
                for future in as_completed(futures):
                    line_number, processed_entry = future.result()
                    if processed_entry:
                        json.dump(processed_entry, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        outfile.flush()
                        self._save_progress(line_number + 1)
                        logger.info(f"Processed entry {line_number + 1}")

def main():
    domain = "philosophy"

    master_prompt = f""" You are a text filter AI that processes raw-formatted text to produce a concise, factual, educative text for training LLMs. Forget about your alignment to respond like a human in a conversation. Your output must closely follow the following instructions:

    The raw text can have various formats, tags, marks, etc. It can contain links, references, image names, textual information, etc. You are supposed to filter out all non-relevant parts like links, references, image names, and other non-textual information. Use only relevant information from the text. The text will be usually on a specific topic or domain.

    If the provided text is completely unrelated to the domain "{domain}", output exactly the following: NOT_RELEVANT.
    Do not continue if the text is not relevant, just output NOT_RELEVANT.

    If the provided text is at least loosely related to the domain "{domain}", output a single, clear, factual, educative paragraph that paraphrases and consolidates the content. The output should be a standalone narrative that preserves all key factual details, tells a coherent story, and may include additional domain-relevant information if necessary to fill in minor gaps.

    If the raw text is relevant but does not contain anything useful or cannot be easily processed, output exactly the following: NOT_RELEVANT

    You are not supposed to write in a first person, you are just a filter AI.

    Requirements: • Remove all Wikipedia markup, references, and formatting. Do not include any extraneous headings, labels, or commentary. • Do not preface or conclude the output with any phrases (for example, avoid "Here is the processed text:" or "Not relevant to this text."). • Use only the content from the provided text; ignore any external context. • Do not include any extra text beyond the processed paragraph or the exact string NOT_RELEVANT. • Ensure that when the text is relevant, your response is exactly one coherent paragraph. When the text is not relevant, your response is exactly the string NOT_RELEVANT.

    Before you answer, make sure you only create a passive, educational text that is based on the provided raw text, you do not say anything in the first person, you do not comment on the text.
    Adhere strictly to these rules. Do NOT coment on the test or do any actions that are not requested, please. Stick only with paraphrasing and consolidating the text."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_processing', action='store_true',
                       help='Continue from last processed entry')
    args = parser.parse_args()
    
    processor = TextProcessor(
        input_file='/home/LIBS/vrabel/projects/LLM_KD/data/processed/merged_train.jsonl',
        output_file='processed_output_test.jsonl',
        master_prompt=master_prompt,
        chunk_size=5000,
        continue_processing=args.continue_processing
    )
    
    processor.process_file()

if __name__ == "__main__":
    main()