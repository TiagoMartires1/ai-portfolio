from nltk.tokenize import sent_tokenize

def chunk_by_sentences(paragraphs, max_words=250, overlap_sentences=1):
    chunks = []
    chunk_id = 0

    for p_index, item in enumerate(paragraphs):
        text = item["text"]
        team = item["team"]

        sentences = sent_tokenize(text)

        current_chunk = []
        current_word_count = 0

        for i, sentence in enumerate(sentences):
            sentence_word_count = len(sentence.split())

            if current_word_count + sentence_word_count <= max_words:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                # Save chunk
                chunks.append({
                    "chunk_id": chunk_id,
                    "team": team,
                    "paragraph_index": p_index,
                    "text": " ".join(current_chunk)
                })
                chunk_id += 1

                # Start new chunk with overlap
                overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk = overlap + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)

        # Save remaining chunk
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "team": team,
                "paragraph_index": p_index,
                "text": " ".join(current_chunk)
            })
            chunk_id += 1

    return chunks