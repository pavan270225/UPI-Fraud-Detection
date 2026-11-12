"""
Screenshot parser module for UPI Fraud Detection.

Extracts transaction features from PhonePe and Google Pay payment screenshots
using OCR and parses them into model-ready feature vectors.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import re
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    Image = None


class ScreenshotParserError(Exception):
    """Raised when screenshot parsing fails."""
    pass


@dataclass
class ParsedTransaction:
    """Parsed transaction data from screenshot."""
    app_type: str  # 'phonepe' or 'gpay'
    amount: Optional[float] = None
    sender_name: Optional[str] = None
    sender_upi: Optional[str] = None
    receiver_name: Optional[str] = None
    receiver_upi: Optional[str] = None
    transaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    status: Optional[str] = None
    bank_name: Optional[str] = None
    raw_text: str = ""

    def to_features(self, target_col: str = 'isFraud', require_target: bool = False) -> pd.DataFrame:
        """Convert parsed transaction to model-ready feature DataFrame."""
        features = {
            'TransactionAmt': [self.amount] if self.amount else [0.0],
            'ProductCD': ['unknown'],
            'card1': [self._hash_string(self.sender_upi or '')],
            'card2': [self._hash_string(self.receiver_upi or '')],
            'card3': [0.0],
            'card4': ['unknown'],
            'card5': [0.0],
            'card6': ['unknown'],
            'addr1': [self._hash_string(self.sender_name or '')],
            'addr2': [self._hash_string(self.receiver_name or '')],
            'dist1': [0.0],
            'dist2': [0.0],
            'P_emaildomain': [self._extract_domain(self.sender_upi or '')],
            'R_emaildomain': [self._extract_domain(self.receiver_upi or '')],
        }

        # Add time-based features if timestamp available
        if self.timestamp:
            try:
                ts = pd.to_datetime(self.timestamp)
                features['TransactionDT'] = [int(ts.timestamp())]
                features['hour'] = [ts.hour]
                features['day_of_week'] = [ts.dayofweek]
            except Exception:
                features['TransactionDT'] = [0]
                features['hour'] = [0]
                features['day_of_week'] = [0]
        else:
            features['TransactionDT'] = [0]
            features['hour'] = [0]
            features['day_of_week'] = [0]

        df = pd.DataFrame(features)

        # Add target column if required (for training, not for inference)
        if require_target:
            df[target_col] = 0  # Default to non-fraud for inference

        return df

    def _hash_string(self, s: str) -> int:
        """Simple hash for string to numeric feature."""
        if not s:
            return 0
        return abs(hash(s)) % 1000000

    def _extract_domain(self, upi: str) -> str:
        """Extract domain from UPI ID."""
        if '@' in upi:
            return upi.split('@')[-1]
        return 'unknown'


class ScreenshotParser:
    """Parse payment screenshots from PhonePe and Google Pay."""

    # Regex patterns for PhonePe
    PHONEPE_PATTERNS = {
        'amount': [
            r'₹\s*([\d,]+\.?\d*)',
            r'Rs\.?\s*([\d,]+\.?\d*)',
            r'INR\s*([\d,]+\.?\d*)',
        ],
        'sender_name': [
            r'From\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Sent by\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Paid by\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
        ],
        'sender_upi': [
            r'(?:From|Sent by|Paid by)\s*[:\-]?\s*[A-Za-z\s]*\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'From\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'Sent by\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
        ],
        'receiver_name': [
            r'To\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Paid to\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Received by\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
        ],
        'receiver_upi': [
            r'(?:To|Paid to|Received by)\s*[:\-]?\s*[A-Za-z\s]*\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'To\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'Paid to\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
        ],
        'transaction_id': [
            r'Transaction ID\s*[:\-]?\s*([A-Z0-9]+)',
            r'Txn ID\s*[:\-]?\s*([A-Z0-9]+)',
            r'Ref(?:erence)?\s*[:\-]?\s*([A-Z0-9]+)',
        ],
        'timestamp': [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)',
            r'(\d{1,2}\s+\w+\s+\d{4}[,\s]+\d{1,2}:\d{2})',
        ],
        'status': [
            r'(Completed|Successful|Failed|Pending|Success)',
        ],
        'bank': [
            r'(?:Bank|via)\s*[:\-]?\s*([A-Za-z\s]+Bank)',
            r'(HDFC|ICICI|SBI|Axis|Kotak|PNB|BOB|Canara|Union|Indian|Yes|IDFC|Federal|IndusInd|RBL|South Indian|Karnataka|Tamilnad|Karur|City Union|Dhanlaxmi|Lakshmi Vilas|J&K|CSB|DBS|Bandhan|IDBI)',
        ],
    }

    # Regex patterns for Google Pay
    GPAY_PATTERNS = {
        'amount': [
            r'₹\s*([\d,]+\.?\d*)',
            r'Rs\.?\s*([\d,]+\.?\d*)',
        ],
        'sender_name': [
            r'From\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Sent by\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
        ],
        'sender_upi': [
            r'(?:From|Sent by)\s*[:\-]?\s*[A-Za-z\s]*\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'From\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'Sent by\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
        ],
        'receiver_name': [
            r'To\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
            r'Paid to\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)',
        ],
        'receiver_upi': [
            r'(?:To|Paid to)\s*[:\-]?\s*[A-Za-z\s]*\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'To\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
            r'Paid to\s*[:\-]?\s*[A-Za-z\s]*\n\s*UPI\s*[:\-]?\s*([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-]+)',
        ],
        'transaction_id': [
            r'Transaction ID\s*[:\-]?\s*([A-Z0-9]+)',
            r'GPAY[:\-]?\s*([A-Z0-9]+)',
            r'Reference\s*[:\-]?\s*([A-Z0-9]+)',
        ],
        'timestamp': [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)',
            r'(\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M)',
        ],
        'status': [
            r'(Completed|Successful|Failed|Pending|Success)',
        ],
        'bank': [
            r'(?:Bank|via)\s*[:\-]?\s*([A-Za-z\s]+Bank)',
            r'(HDFC|ICICI|SBI|Axis|Kotak|PNB|BOB|Canara|Union|Indian|Yes|IDFC|Federal|IndusInd|RBL)',
        ],
    }

    def __init__(self):
        if not OCR_AVAILABLE:
            raise ScreenshotParserError(
                "OCR dependencies not installed. Run: pip install pytesseract pillow"
            )

    def detect_app(self, text: str) -> str:
        """Detect if screenshot is from PhonePe or Google Pay."""
        text_lower = text.lower()
        # Check for PhonePe first
        if any(kw in text_lower for kw in ['phonepe', 'phone pe']):
            return 'phonepe'
        # Check for Google Pay variants (more specific matches)
        elif any(kw in text_lower for kw in ['google pay', 'gpay', 'googlepay']):
            return 'gpay'
        elif 'tez' in text_lower and 'payment' in text_lower:
            return 'gpay'
        # Default to phonepe if unclear
        return 'phonepe'

    def extract_text(self, image_path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(image_path)
            # Enhance image for better OCR
            image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise ScreenshotParserError(f"OCR failed: {e}")

    def parse(self, image_path: Path) -> ParsedTransaction:
        """Parse a payment screenshot and extract transaction features."""
        raw_text = self.extract_text(image_path)
        app_type = self.detect_app(raw_text)

        patterns = self.PHONEPE_PATTERNS if app_type == 'phonepe' else self.GPAY_PATTERNS

        transaction = ParsedTransaction(
            app_type=app_type,
            raw_text=raw_text
        )

        # Extract each field using patterns
        transaction.amount = self._extract_amount(raw_text, patterns)
        transaction.sender_name = self._extract_first_match(raw_text, patterns.get('sender_name', []))
        transaction.sender_upi = self._extract_first_match(raw_text, patterns.get('sender_upi', []))
        transaction.receiver_name = self._extract_first_match(raw_text, patterns.get('receiver_name', []))
        transaction.receiver_upi = self._extract_first_match(raw_text, patterns.get('receiver_upi', []))
        transaction.transaction_id = self._extract_first_match(raw_text, patterns.get('transaction_id', []))
        transaction.timestamp = self._extract_first_match(raw_text, patterns.get('timestamp', []))
        transaction.status = self._extract_first_match(raw_text, patterns.get('status', []))
        transaction.bank_name = self._extract_first_match(raw_text, patterns.get('bank', []))

        return transaction

    def _extract_amount(self, text: str, patterns: Dict) -> Optional[float]:
        """Extract and parse amount from text."""
        for pattern in patterns.get('amount', []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the largest amount (usually the transaction amount)
                amounts = [float(m.replace(',', '')) for m in matches]
                return max(amounts) if amounts else None
        return None

    def _extract_first_match(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract first matching pattern from text."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


def parse_screenshot(image_path: str) -> ParsedTransaction:
    """
    Convenience function to parse a payment screenshot.

    Args:
        image_path: Path to the screenshot image file

    Returns:
        ParsedTransaction with extracted features
    """
    parser = ScreenshotParser()
    return parser.parse(Path(image_path))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        try:
            result = parse_screenshot(sys.argv[1])
            print(f"App: {result.app_type}")
            print(f"Amount: {result.amount}")
            print(f"Sender: {result.sender_name} ({result.sender_upi})")
            print(f"Receiver: {result.receiver_name} ({result.receiver_upi})")
            print(f"Transaction ID: {result.transaction_id}")
            print(f"Timestamp: {result.timestamp}")
            print(f"Status: {result.status}")
            print(f"Bank: {result.bank_name}")
            print("\nFeatures DataFrame:")
            df = result.to_features()
            print(df.to_string())
        except ScreenshotParserError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python screenshot_parser.py <image_path>")