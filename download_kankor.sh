#!/bin/bash

# ==============================================================================
# Afghanistan MoE Textbook Downloader (Grades 10, 11, 12)
# Source: https://moe.gov.af/dr/كتب-نصاب-تعليمى
# Requirements: curl, wget, grep (all pre-installed on Ubuntu/Debian)
# ==============================================================================

set -e

BASE_URL="https://moe.gov.af"
PAGE_URL="https://moe.gov.af/dr/%D9%83%D8%AA%D8%A8-%D9%86%D8%B5%D8%A7%D8%A8-%D8%AA%D8%B9%D9%84%D9%8A%D9%85%D9%89"
OUTPUT_DIR="data/raw_pdfs"

echo "📁 Creating output folders..."
mkdir -p "$OUTPUT_DIR/grade_10"
mkdir -p "$OUTPUT_DIR/grade_11"
mkdir -p "$OUTPUT_DIR/grade_12"
mkdir -p "$OUTPUT_DIR/unsorted"

# ==============================================================================
# STEP 1: Scrape the page and extract ALL PDF/file URLs
# ==============================================================================

echo "🌐 Fetching the MoE textbook page..."
curl -s -A "Mozilla/5.0 (X11; Linux x86_64)" "$PAGE_URL" -o /tmp/moe_page.html

echo "🔍 Extracting PDF and file links..."

# Extract all href links that point to files (pdf, doc, etc.)
grep -oP 'href="[^"]*\.(pdf|PDF)[^"]*"' /tmp/moe_page.html \
    | sed 's/href="//g' \
    | sed 's/"$//g' \
    | sort -u > /tmp/pdf_links.txt

# Also catch relative paths (starting with /)
grep -oP "href='/[^']*\.(pdf|PDF)[^']*'" /tmp/moe_page.html \
    | sed "s/href='//g" \
    | sed "s/'$//g" \
    | sort -u >> /tmp/pdf_links.txt

# Prepend base URL to any relative paths
awk -v base="$BASE_URL" '{
    if ($0 ~ /^http/) print $0;
    else print base $0
}' /tmp/pdf_links.txt | sort -u > /tmp/final_links.txt

TOTAL=$(wc -l < /tmp/final_links.txt)
echo "✅ Found $TOTAL PDF links on the page."

# ==============================================================================
# STEP 2: Download each PDF into the correct grade folder
# ==============================================================================

while IFS= read -r url; do
    FILENAME=$(basename "$url" | sed 's/%20/ /g' | sed 's/%[0-9A-Fa-f]\{2\}//g')
    
    # Route into correct grade folder based on keywords in URL
    if echo "$url" | grep -qi "G10\|grade.10\|صنف.ده\|10th"; then
        DEST="$OUTPUT_DIR/grade_10"
    elif echo "$url" | grep -qi "G11\|grade.11\|یازدهم\|11th"; then
        DEST="$OUTPUT_DIR/grade_11"
    elif echo "$url" | grep -qi "G12\|grade.12\|دوازدهم\|12th"; then
        DEST="$OUTPUT_DIR/grade_12"
    else
        DEST="$OUTPUT_DIR/unsorted"
    fi

    echo "⬇️  Downloading: $FILENAME → $DEST"
    wget -q --show-progress -c -P "$DEST" "$url" || echo "  ⚠️  Failed: $url"

done < /tmp/final_links.txt

# ==============================================================================
# STEP 3: Print summary
# ==============================================================================

echo ""
echo "============================================"
echo "✅ Download complete. Summary:"
echo "  Grade 10: $(ls $OUTPUT_DIR/grade_10/*.pdf 2>/dev/null | wc -l) files"
echo "  Grade 11: $(ls $OUTPUT_DIR/grade_11/*.pdf 2>/dev/null | wc -l) files"
echo "  Grade 12: $(ls $OUTPUT_DIR/grade_12/*.pdf 2>/dev/null | wc -l) files"
echo "  Unsorted: $(ls $OUTPUT_DIR/unsorted/*.pdf 2>/dev/null | wc -l) files"
echo "  Saved to: $OUTPUT_DIR"
echo "============================================"
