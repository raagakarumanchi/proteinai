import requests
import pandas as pd
from typing import Dict, List, Optional
import time
from tqdm import tqdm
import logging
from functools import lru_cache
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class UniProtClient:
    """Client for fetching protein data from UniProt API"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.base_url = "https://rest.uniprot.org"
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session with timeout and retry
        self.session.timeout = 30
        self.max_retries = 3
    
    def _get_cache_path(self, query: str, limit: int) -> Path:
        """Generate cache file path"""
        import hashlib
        cache_key = hashlib.md5(f"{query}_{limit}".encode()).hexdigest()
        return self.cache_dir / f"uniprot_{cache_key}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent"""
        if not cache_path.exists():
            return None
        
        # Check if cache is recent (less than 24 hours old)
        import time
        if time.time() - cache_path.stat().st_mtime > 86400:
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache file corrupted: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path):
        """Save data to cache"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data.to_dict('records'), f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def search_proteins(self, query: str, limit: int = 100) -> pd.DataFrame:
        """Search for proteins by query string with caching"""
        cache_path = self._get_cache_path(query, limit)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            logger.info(f"Loaded {len(cached_data)} proteins from cache")
            return cached_data
        
        url = f"{self.base_url}/uniprotkb/search"
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,id,protein_name,organism_name,length,sequence",
            "size": limit
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse TSV response
                lines = response.text.strip().split('\n')
                if len(lines) <= 1:
                    return pd.DataFrame()
                
                data = []
                for line in lines[1:]:  # Skip header
                    fields = line.split('\t')
                    if len(fields) >= 6:
                        data.append({
                            'accession': fields[0],
                            'id': fields[1],
                            'protein_name': fields[2],
                            'organism': fields[3],
                            'length': int(fields[4]) if fields[4].isdigit() else 0,
                            'sequence': fields[5]
                        })
                
                df = pd.DataFrame(data)
                
                # Save to cache
                self._save_to_cache(df, cache_path)
                logger.info(f"Fetched and cached {len(df)} proteins")
                
                return df
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch data after {self.max_retries} attempts")
                    return pd.DataFrame()
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_protein_details(self, accession: str) -> Dict:
        """Get detailed information for a specific protein"""
        url = f"{self.base_url}/uniprotkb/{accession}"
        params = {"format": "json"}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching protein details: {e}")
            return {}
    
    def get_evolutionary_data(self, protein_family: str) -> pd.DataFrame:
        """Get evolutionary data for a protein family"""
        query = f"family:{protein_family}"
        return self.search_proteins(query, limit=200)
    
    def batch_fetch_sequences(self, accessions: List[str]) -> Dict[str, str]:
        """Fetch sequences for multiple proteins"""
        sequences = {}
        
        for accession in tqdm(accessions, desc="Fetching sequences"):
            details = self.get_protein_details(accession)
            if details and 'sequence' in details:
                sequences[accession] = details['sequence']['value']
            time.sleep(0.1)  # Rate limiting
        
        return sequences
