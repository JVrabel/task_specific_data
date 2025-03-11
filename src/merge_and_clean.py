import json
import re

def merge_and_clean_jsonl_files(file1, file2, output_file):
    # Set to keep track of seen texts to avoid duplicates
    seen_texts = set()
    entries_written = 0
    entries_cleaned = 0
    entries_removed = 0
    
    def should_remove_entry(text):
        # Remove entry if it matches any case version of "NOT_RELEVANT."
        return text.strip().lower() == "not_relevant."
    
    def clean_text(text):
        # Remove any case version of "NOT_RELEVANT" from the text
        pattern = re.compile(r'not_relevant', re.IGNORECASE)
        return pattern.sub('', text).strip()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process first file
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in f1:
                entry = json.loads(line)
                
                # Check if entry should be completely removed
                if should_remove_entry(entry['text']):
                    entries_removed += 1
                    continue
                
                # Remove any case version of "NOT_RELEVANT" from text if present
                if re.search(r'not_relevant', entry['text'], re.IGNORECASE):
                    entry['text'] = clean_text(entry['text'])
                    entries_cleaned += 1
                
                if entry['text'] and entry['text'] not in seen_texts:
                    seen_texts.add(entry['text'])
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    entries_written += 1
        
        # Process second file
        with open(file2, 'r', encoding='utf-8') as f2:
            for line in f2:
                entry = json.loads(line)
                
                # Check if entry should be completely removed
                if should_remove_entry(entry['text']):
                    entries_removed += 1
                    continue
                
                # Remove any case version of "NOT_RELEVANT" from text if present
                if re.search(r'not_relevant', entry['text'], re.IGNORECASE):
                    entry['text'] = clean_text(entry['text'])
                    entries_cleaned += 1
                
                if entry['text'] and entry['text'] not in seen_texts:
                    seen_texts.add(entry['text'])
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    entries_written += 1
    
    print(f"Total entries written: {entries_written}")
    print(f"Entries cleaned (NOT_RELEVANT removed): {entries_cleaned}")
    print(f"Entries removed (exactly NOT_RELEVANT.): {entries_removed}")

# Use the function
merge_and_clean_jsonl_files(
    'task_specific_data/processed_output_test.jsonl',
    'LLM_KD/data/gutenberg/train/bookshelf_57_train.jsonl',
    'merged_clean_output.jsonl'
)