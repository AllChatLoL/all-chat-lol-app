import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import time
import random
import json
import os
import shutil

# Configure pandas to use the new behavior for fillna operations
pd.set_option('future.no_silent_downcasting', True)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Authenticate with Google Sheets
creds_dict = st.secrets["gcp_service_account"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
client = gspread.authorize(creds)

# League sheet names
leagues = {
    "LCK Cup 2025": "LCK Cup 2025 Data",
    "LEC Winter 2025": "LEC Winter 2025 Data",
    "LPL Winter 2025": "LPL Winter 2025 Data",
    "LTA North Winter 2025": "LTA North Winter 2025 Data",
    "LCP Winter 2025": "LCP Winter 2025 Data",
}

roles = ["Top", "Jungle", "Mid", "Bot", "Support"]

# Expected column names for players
player_columns = {
    "Player": "Player",
    "Win Rate": "Win Rate",
    "Gold Per Minute": "Gold Per Minute",
    "Damage Per Minute": "Damage Per Minute",
    "KDA": "KDA",
    "Kill Participation": "Kill Participation",
    "Gold Percentage": "Gold Percentage",
    "CS Per Minute": "CS Per Minute",
    "Score": "Score"
}

# Expected column names for teams
team_columns = {
    "Team": "Team",
    "Score": "Score",
    "Win Rate": "Win Rate",
    "Gold Per Minute": "Gold Per Minute",
    "Kills/Game": "Kills/Game",
    "Deaths/Game": "Deaths/Game",
    "DPM": "DPM",
    "Gold Diff @ 15": "Gold Diff @ 15",
    "Tower Diff": "Tower Diff"
}

# Function to clean player names
def clean_player_name(name):
    return re.sub(r"^\d+\.\s*", "", name).strip()

# Function to extract numeric score for sorting
def extract_score(score_str):
    try:
        return float(score_str)
    except (ValueError, TypeError):
        return 0.0

# Cache the Google Sheets data - this will only fetch each sheet once per session
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_raw_sheet_data(sheet_name, tab_name):
    """Fetch raw data from Google Sheet and cache it with rate limiting and retries"""
    max_retries = 5  # Increased from 3 to 5
    retry_count = 0
    
    # Add throttling to limit overall request rate
    # This ensures we don't exceed Google's rate limits
    if 'last_api_call_time' in st.session_state:
        # Ensure at least 1.5 seconds between ANY API calls
        elapsed = time.time() - st.session_state['last_api_call_time']
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
    
    # Update the last API call time
    st.session_state['last_api_call_time'] = time.time()
    
    # Check if we have the data cached on disk to avoid API call completely
    cache_dir = ".sheet_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{sheet_name}_{tab_name}.json")
    
    # Try to load from disk cache first
    try:
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:  # 24hr cache
            with open(cache_file, 'r') as f:
                print(f"Loading {sheet_name}, {tab_name} from disk cache")
                cached_data = json.load(f)
                return cached_data
    except Exception as e:
        print(f"Error loading from disk cache: {e}")
        # Continue to normal fetch if cache fails
    
    while retry_count < max_retries:
        try:
            # Add random delay to avoid hitting rate limits
            delay = random.uniform(1.0, 2.5) * (retry_count + 1)  # Progressive delay
            print(f"Waiting {delay:.2f}s before fetching {sheet_name}, {tab_name}")
            time.sleep(delay)
            
            # Try a different approach on each retry to maximize chances of success
            if retry_count == 0:
                # First attempt: Standard approach
                print(f"Fetching {sheet_name}, {tab_name} using standard method")
                sheet = client.open(sheet_name).worksheet(tab_name)
                data = sheet.get_all_values()
            else:
                # Alternative approach for retries
                print(f"Fetching {sheet_name}, {tab_name} using alternative method (attempt {retry_count+1})")
                # Try to directly use the spreadsheet ID and range
                try:
                    # Get the spreadsheet ID first
                    spreadsheet = client.open(sheet_name)
                    spreadsheet_id = spreadsheet.id
                    
                    # Get the worksheet ID
                    worksheet = spreadsheet.worksheet(tab_name)
                    
                    # Now use the values_get method directly
                    result = client.values_get(
                        spreadsheetId=spreadsheet_id,
                        range=f"'{tab_name}'!A1:Z1000"  # Use a large range to get all data
                    )
                    data = result.get('values', [])
                    print(f"Successfully retrieved data using alternative method: {len(data)} rows")
                except Exception as alt_err:
                    print(f"Alternative method failed: {alt_err}")
                    # Fall back to standard method
                    sheet = client.open(sheet_name).worksheet(tab_name)
                    data = sheet.get_all_values()
            
            # Fix for Response objects with 200 status - this suggests the API call succeeded
            # but the data isn't being extracted properly
            if hasattr(data, 'status_code') and data.status_code == 200:
                print(f"Got Response object with 200 status for {sheet_name}, {tab_name}. Extracting data...")
                try:
                    # Try to extract data from the response based on gspread response format
                    if hasattr(data, 'json'):
                        json_data = data.json()
                        if 'values' in json_data:
                            data = json_data['values']
                        else:
                            print("Response has no 'values' key in JSON")
                    else:
                        print("Response has no JSON method")
                        
                    # If we still have a Response object, try one more approach
                    if hasattr(data, 'status_code'):
                        # Try to get the spreadsheet ID and use values_get directly
                        try:
                            spreadsheet = client.open(sheet_name)
                            spreadsheet_id = spreadsheet.id
                            result = client.values_get(
                                spreadsheetId=spreadsheet_id,
                                range=f"'{tab_name}'!A1:Z1000"
                            )
                            data = result.get('values', [])
                        except Exception as direct_err:
                            print(f"Direct API call failed: {direct_err}")
                            
                except Exception as extract_error:
                    print(f"Error extracting data from response: {extract_error}")
            
            # Verify we got actual data and not just a response object
            if isinstance(data, list) and len(data) > 0:
                print(f"Successfully retrieved {len(data)} rows of data for {sheet_name}, {tab_name}")
                # Save to disk cache for future use
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                    print(f"Saved {sheet_name}, {tab_name} to disk cache")
                except Exception as cache_err:
                    print(f"Error saving to disk cache: {cache_err}")
                    
                return data
            else:
                print(f"Warning: Empty or invalid data returned for {sheet_name}, {tab_name}")
                print(f"Data type: {type(data)}")
                
                # If data is a Response object, print more details
                if hasattr(data, 'status_code'):
                    print(f"Response status: {data.status_code}")
                    try:
                        print(f"Response content (first 200 chars): {str(data.content)[:200]}")
                    except:
                        print("Could not access response content")
                
                # If data is not what we expect, try again
                retry_count += 1
                time.sleep(2 * retry_count)  # Progressively longer waits
                continue
                
        except Exception as e:
            retry_count += 1
            error_message = str(e)
            
            # If rate limit error, wait longer before retrying
            if "429" in error_message or "Quota exceeded" in error_message:
                wait_time = min(60, 4 ** retry_count)  # Cap at 60 seconds but use exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            elif retry_count < max_retries:
                # For other errors, wait a shorter time
                wait_time = 2 * retry_count
                print(f"Error fetching {sheet_name}, {tab_name}: {error_message}. Retry {retry_count}/{max_retries} in {wait_time}s")
                time.sleep(wait_time)
            
            # If all retries exhausted, return the error
            if retry_count == max_retries:
                return f"Error: {error_message}"
    
    # If we've exhausted retries without getting valid data
    return f"Error: Could not retrieve valid data for {sheet_name}, {tab_name} after {max_retries} attempts"

# New function to fetch all tabs from a sheet in a single call
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_all_sheet_tabs(sheet_name):
    """Fetch all tabs from a single sheet at once to minimize API calls"""
    max_retries = 5
    retry_count = 0
    
    # Add throttling to limit overall request rate
    if 'last_api_call_time' in st.session_state:
        # Ensure at least 1.5 seconds between ANY API calls
        elapsed = time.time() - st.session_state['last_api_call_time']
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
    
    # Update the last API call time
    st.session_state['last_api_call_time'] = time.time()
    
    # Check if we have the data cached on disk to avoid API call completely
    cache_dir = ".sheet_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{sheet_name}_ALL_TABS.json")
    
    # Try to load from disk cache first
    try:
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:  # 24hr cache
            with open(cache_file, 'r') as f:
                print(f"Loading all tabs for {sheet_name} from disk cache")
                cached_data = json.load(f)
                return cached_data
    except Exception as e:
        print(f"Error loading from disk cache: {e}")
    
    while retry_count < max_retries:
        try:
            # Add random delay to avoid hitting rate limits
            delay = random.uniform(1.0, 2.5) * (retry_count + 1)
            print(f"Waiting {delay:.2f}s before fetching all tabs from {sheet_name}")
            time.sleep(delay)
            
            # Open the spreadsheet
            spreadsheet = client.open(sheet_name)
            
            # Get all worksheets at once - this is ONE API call
            all_worksheets = spreadsheet.worksheets()
            print(f"Retrieved {len(all_worksheets)} worksheets from {sheet_name}")
            
            # Fetch data for each worksheet - this is a separate API call for each tab
            # but we're getting all of them at once and caching them
            all_data = {}
            
            # Loop through all worksheets
            for worksheet in all_worksheets:
                tab_name = worksheet.title
                print(f"Getting data for tab: {tab_name}")
                
                # Check if this tab is relevant to our application
                if tab_name in roles or tab_name == "Power Rankings":
                    try:
                        # Get all values for this worksheet
                        tab_data = worksheet.get_all_values()
                        all_data[tab_name] = tab_data
                        print(f"Successfully fetched {len(tab_data)} rows for {tab_name}")
                    except Exception as tab_error:
                        print(f"Error fetching data for tab {tab_name}: {str(tab_error)}")
                        all_data[tab_name] = f"Error: {str(tab_error)}"
                        # Continue with other tabs instead of failing completely
            
            # Save to disk cache for future use
            try:
                with open(cache_file, 'w') as f:
                    json.dump(all_data, f)
                print(f"Saved all tabs for {sheet_name} to disk cache")
            except Exception as cache_err:
                print(f"Error saving to disk cache: {cache_err}")
            
            # Return the data if we have at least some tabs
            if all_data:
                return all_data
            
            # If we get here, we couldn't fetch any tabs
            retry_count += 1
            time.sleep(2 * retry_count)
            
        except Exception as e:
            retry_count += 1
            error_message = str(e)
            
            # If rate limit error, wait longer before retrying
            if "429" in error_message or "Quota exceeded" in error_message:
                wait_time = min(60, 4 ** retry_count)  # Cap at 60 seconds but use exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            elif retry_count < max_retries:
                # For other errors, wait a shorter time
                wait_time = 2 * retry_count
                print(f"Error fetching all tabs from {sheet_name}: {error_message}. Retry {retry_count}/{max_retries} in {wait_time}s")
                time.sleep(wait_time)
            
            # If all retries exhausted, return the error
            if retry_count == max_retries:
                return f"Error: {error_message}"
    
    return f"Error: Could not retrieve valid data for {sheet_name} after {max_retries} attempts"

# Process player data after fetching (all data transformation operations)
def process_player_data(data, tab_name):
    """Process raw sheet data into player rankings dataframe"""
    if isinstance(data, str):  # Error message
        return data
        
    if not data:
        return f"No data found for {tab_name}"
        
    df = pd.DataFrame(data)
    
    if df.empty:
        return f"Empty dataframe for {tab_name}"
        
    df.columns = df.iloc[0]  # Set first row as column names
    df = df[1:]  # Remove first row from data
    df = df.reset_index(drop=True)

    # Extract player stats (Columns A-H) and power rankings (Columns I-J)
    player_stats = df.iloc[:, :8].copy()  # A-H (Stats) - create explicit copy
    power_scores = df.iloc[:, 8:10].copy()  # I-J (Power Rankings) - create explicit copy
    
    # Ensure columns exist
    if power_scores.shape[1] >= 2:
        power_scores.columns = ["Player", "Score"]
        
        # Fix SettingWithCopyWarning by using .loc
        power_scores.loc[:, "Player"] = power_scores["Player"].apply(clean_player_name)
        power_scores.loc[:, "Score"] = power_scores["Score"].apply(extract_score)

        # Merge by Player Name (Exact Match)
        final_df = pd.merge(power_scores, player_stats, on="Player", how="left")

        # Ensure sorting by Power Score (Descending)
        final_df = final_df.sort_values(by="Score", ascending=False)
        return final_df
    else:
        return f"Power score columns not found for {tab_name}"

# Process team ranking data after fetching
def process_team_data(data, sheet_name):
    """Process raw sheet data into team rankings dataframe"""
    if isinstance(data, str):  # Error message
        return data
        
    # Check for empty or invalid data responses
    if not data or not isinstance(data, list):
        return f"No valid data found for '{sheet_name}'"
        
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            return f"Empty dataframe for '{sheet_name}'"
            
        # Print the first few rows for debugging
        print(f"First 3 rows of raw data for {sheet_name}:")
        for i in range(min(3, len(data))):
            print(data[i])
            
        # Check if we have enough rows to work with
        if len(df) <= 1:
            return f"Not enough rows in data for '{sheet_name}'"
            
        # Set column names from first row
        df.columns = df.iloc[0]  # Set first row as column names
        df = df[1:]  # Remove first row from data
        df = df.reset_index(drop=True)
        
        # Debug column names
        print(f"Column names for {sheet_name}: {df.columns.tolist()}")
        
        # Ensure required columns exist
        if "Score" not in df.columns:
            print(f"Warning: 'Score' column not found in {sheet_name}. Columns: {df.columns.tolist()}")
            # Try to find Score column with different casing
            score_col = next((col for col in df.columns if col.lower() == "score"), None)
            if score_col:
                df = df.rename(columns={score_col: "Score"})
            else:
                # Add dummy Score column
                df["Score"] = 100
        
        if "Team" not in df.columns:
            print(f"Warning: 'Team' column not found in {sheet_name}. Columns: {df.columns.tolist()}")
            # Try to find Team column with different casing
            team_col = next((col for col in df.columns if col.lower() == "team"), None)
            if team_col:
                df = df.rename(columns={team_col: "Team"})
            else:
                # Return error if we can't find a team column at all
                return f"Required 'Team' column not found in {sheet_name}"
        
        # Remove numbers from team names
        df.loc[:, "Team"] = df["Team"].apply(lambda x: re.sub(r"^\d+\.\s*", "", str(x)))

        # Convert Score to numeric for proper sorting
        df.loc[:, "Score"] = pd.to_numeric(df["Score"], errors="coerce")
        
        # Fill NA values
        df.loc[:, "Score"] = df["Score"].fillna(0).infer_objects(copy=False)
        
        # Sort by Score (Descending)
        df = df.sort_values(by="Score", ascending=False)

        return df
    except Exception as e:
        print(f"Error processing team data for {sheet_name}: {str(e)}")
        return f"Error processing data: {str(e)}"

# Cache the processed player data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_player_data(sheet_name, role):
    """Get processed player data with caching"""
    raw_data = fetch_raw_sheet_data(sheet_name, role)
    return process_player_data(raw_data, role)

# Cache the processed team data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_team_data(sheet_name):
    """Get processed team data with caching"""
    raw_data = fetch_raw_sheet_data(sheet_name, "Power Rankings")
    return process_team_data(raw_data, sheet_name)

# Load all data at startup to fill the cache
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_all_data():
    """Preload all data into cache at startup with rate limiting"""
    all_data = {
        "player_data": {},
        "team_data": {}
    }
    
    # Create a progress bar
    progress_text = "Loading data from all leagues (this may take a moment due to API rate limits)..."
    progress_bar = st.progress(0.0, text=progress_text)
    
    # Count total operations for progress bar
    total_operations = len(leagues) * (len(roles) + 1)  # roles + team data per league
    completed_operations = 0
    
    # Fetch team data first (most important)
    for league_name, sheet_name in leagues.items():
        all_data["team_data"][league_name] = get_team_data(sheet_name)
        
        # Update progress
        completed_operations += 1
        progress_bar.progress(completed_operations / total_operations, 
                             text=f"{progress_text} Loading {league_name} team data...")
        
        # Small delay between leagues to avoid rate limits
        time.sleep(0.5)
    
    # Fetch player data by role
    for league_name, sheet_name in leagues.items():
        all_data["player_data"][league_name] = {}
        
        for role in roles:
            all_data["player_data"][league_name][role] = get_player_data(sheet_name, role)
            
            # Update progress
            completed_operations += 1
            progress_bar.progress(completed_operations / total_operations, 
                                 text=f"{progress_text} Loading {league_name} {role} data...")
            
            # Small delay between requests to avoid rate limits
            time.sleep(0.5)
    
    # Clear the progress bar
    progress_bar.empty()
    
    return all_data

# Helper functions for worker-based data loading
def fetch_and_process_team_data(sheet_name, league_name):
    """Worker function to fetch and process team data"""
    try:
        raw_data = fetch_raw_sheet_data(sheet_name, "Power Rankings")
        # Apply random delay for rate limit mitigation
        time.sleep(random.uniform(0.5, 1.5))
        return process_team_data(raw_data, sheet_name)
    except Exception as e:
        print(f"Error in worker fetching team data for {league_name}: {str(e)}")
        return f"Error: {str(e)}"

def fetch_and_process_player_data(sheet_name, role, league_name):
    """Worker function to fetch and process player data"""
    try:
        raw_data = fetch_raw_sheet_data(sheet_name, role)
        # Apply random delay for rate limit mitigation
        time.sleep(random.uniform(0.5, 1.5))
        return process_player_data(raw_data, role)
    except Exception as e:
        print(f"Error in worker fetching player data for {league_name} {role}: {str(e)}")
        return f"Error: {str(e)}"

# Optimized data loading with worker threads
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_all_data_with_workers():
    """Preload all data into cache at startup using worker threads for efficiency"""
    all_data = {
        "player_data": {},
        "team_data": {}
    }
    
    progress_text = "Loading data from all leagues using parallel workers..."
    progress_bar = st.progress(0.0, text=progress_text)
    
    # Track all futures and their metadata
    futures = []
    total_operations = len(leagues) * (len(roles) + 1)
    completed_operations = 0
    
    # Create a thread pool with VERY limited workers to avoid hitting rate limits
    with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced from 3 to 2
        # First submit team data tasks (higher priority)
        team_futures = []
        for league_name, sheet_name in leagues.items():
            all_data["team_data"][league_name] = None
            future = executor.submit(fetch_and_process_team_data, sheet_name, league_name)
            team_futures.append((future, league_name))
            futures.append((future, "team", league_name, None))
            
            # Add a delay between submissions to spread out requests
            time.sleep(1.0)  # 1 second delay between submissions
        
        # Wait for team data to complete before fetching player data
        # This ensures we have the most critical data first
        for future, league_name in team_futures:
            try:
                result = future.result()
                all_data["team_data"][league_name] = result
                completed_operations += 1
                progress_bar.progress(completed_operations / total_operations,
                                    text=f"{progress_text} Completed team data for {league_name}...")
            except Exception as e:
                all_data["team_data"][league_name] = f"Error: {str(e)}"
                completed_operations += 1
                progress_bar.progress(completed_operations / total_operations)
        
        # Then submit player data tasks ONE LEAGUE AT A TIME to avoid overwhelming the API
        for league_name, sheet_name in leagues.items():
            all_data["player_data"][league_name] = {}
            league_player_futures = []
            
            for role in roles:
                all_data["player_data"][league_name][role] = None
                future = executor.submit(fetch_and_process_player_data, sheet_name, role, league_name)
                league_player_futures.append((future, role))
                
                # Add a delay between submissions
                time.sleep(1.5)  # 1.5 second delay between submissions
            
            # Process player results for this league as they complete
            for future, role in league_player_futures:
                try:
                    result = future.result()
                    all_data["player_data"][league_name][role] = result
                except Exception as e:
                    all_data["player_data"][league_name][role] = f"Error: {str(e)}"
                
                # Update progress
                completed_operations += 1
                progress_bar.progress(completed_operations / total_operations,
                                    text=f"{progress_text} Completed {league_name} {role} data...")
            
            # Add a delay between leagues
            time.sleep(2.0)  # 2 second delay between leagues
    
    # Clear the progress bar
    progress_bar.empty()
    return all_data

# Optimized data loading with worker threads - UNIFIED APPROACH
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_all_data_with_workers():
    """Preload all data into cache at startup using worker threads with the unified approach
    that minimizes API calls by fetching all tabs at once per spreadsheet"""
    all_data = {
        "player_data": {},
        "team_data": {}
    }
    
    progress_text = "Loading data from all leagues (optimized method)..."
    progress_bar = st.progress(0.0, text=progress_text)
    
    total_leagues = len(leagues)
    completed_leagues = 0
    
    # Create a thread pool with limited workers to avoid hitting rate limits
    with ThreadPoolExecutor(max_workers=2) as executor:  # Just 2 workers
        # Create futures for each league's sheet (this is much more efficient)
        futures = []
        for league_name, sheet_name in leagues.items():
            all_data["player_data"][league_name] = {}
            all_data["team_data"][league_name] = None
            
            # Add to worker pool - each worker will fetch ALL tabs for one sheet
            future = executor.submit(fetch_all_sheet_tabs, sheet_name)
            futures.append((future, league_name, sheet_name))
            
            # Add a delay between submissions to spread out requests
            time.sleep(1.5)  # 1.5 second delay between submissions
        
        # Process results as they complete
        for future, league_name, sheet_name in futures:
            try:
                # Get the result (all tabs for this sheet)
                all_tabs_data = future.result()
                
                # Check if we got an error
                if isinstance(all_tabs_data, str):
                    print(f"Error fetching all tabs for {league_name}: {all_tabs_data}")
                    all_data["team_data"][league_name] = all_tabs_data
                    for role in roles:
                        all_data["player_data"][league_name][role] = all_tabs_data
                else:
                    # Process team data
                    if "Power Rankings" in all_tabs_data:
                        team_raw_data = all_tabs_data["Power Rankings"]
                        all_data["team_data"][league_name] = process_team_data(team_raw_data, sheet_name)
                    else:
                        all_data["team_data"][league_name] = "Power Rankings tab not found"
                    
                    # Process player data for each role
                    for role in roles:
                        if role in all_tabs_data:
                            player_raw_data = all_tabs_data[role]
                            all_data["player_data"][league_name][role] = process_player_data(player_raw_data, role)
                        else:
                            all_data["player_data"][league_name][role] = f"{role} tab not found"
            
            except Exception as e:
                error_msg = f"Error processing {league_name} data: {str(e)}"
                print(error_msg)
                all_data["team_data"][league_name] = error_msg
                for role in roles:
                    all_data["player_data"][league_name][role] = error_msg
            
            # Update progress
            completed_leagues += 1
            progress_bar.progress(completed_leagues / total_leagues,
                                text=f"{progress_text} Completed {league_name} data ({completed_leagues}/{total_leagues})...")
    
    # Clear the progress bar
    progress_bar.empty()
    return all_data

# Add a function to safely load data with fallbacks for rate limit errors
def safely_load_data():
    """Load data with fallbacks for rate limit errors"""
    try:
        # Try to load all data with workers first (more efficient)
        st.info("Loading data using parallel workers...")
        try:
            return load_all_data_with_workers()
        except Exception as worker_error:
            st.warning(f"Worker-based loading failed: {str(worker_error)}. Falling back to sequential loading.")
            # Fall back to single-threaded loading if worker loading fails
        return load_all_data()
    except Exception as e:
        st.warning(f"Encountered an error while loading all data: {str(e)}")
        st.info("Falling back to loading data as needed. Some features may be slower.")
        
        # Return minimal data structure for lazy loading
        return {
            "player_data": {},
            "team_data": {}
        }

# Add a helper function to get team data with fallback
def get_team_data_with_fallback(league):
    """Get team data with fallback for when it's not in cache"""
    # Check if we already have this data
    if league in st.session_state.all_data["team_data"]:
        cached_data = st.session_state.all_data["team_data"][league]
        # Make sure the cached data is actually valid and not an error message
        if not isinstance(cached_data, str):
            return cached_data
        else:
            print(f"Cached data for {league} is an error message: {cached_data}")
            # Continue to reload data
    
    # Otherwise, load it now
    with st.spinner(f"Loading {league} team data..."):
        try:
            if league not in leagues:
                return f"League '{league}' not found in configured leagues"
                
            print(f"Attempting to fetch fresh data for {league} team rankings")
            sheet_name = leagues[league]
            data = fetch_raw_sheet_data(sheet_name, "Power Rankings")
            
            # If data is a string, it's an error message
            if isinstance(data, str):
                print(f"Error fetching team data for {league}: {data}")
                return data
                
            processed_data = process_team_data(data, sheet_name)
            
            # Store in session state for future use only if it's a valid DataFrame
            if not isinstance(processed_data, str):
                st.session_state.all_data["team_data"][league] = processed_data
                return processed_data
            else:
                print(f"Error processing team data for {league}: {processed_data}")
                return processed_data
                
        except Exception as e:
            error_msg = f"Unexpected error fetching {league} team data: {str(e)}"
            print(error_msg)
            return error_msg

# Updated helper function to get team data with fallback using unified approach
def get_team_data_with_fallback(league):
    """Get team data with fallback using the unified approach that fetches all tabs at once"""
    # Check if we already have this data
    if league in st.session_state.all_data["team_data"]:
        cached_data = st.session_state.all_data["team_data"][league]
        # Make sure the cached data is actually valid and not an error message
        if not isinstance(cached_data, str):
            return cached_data
        else:
            print(f"Cached data for {league} is an error message: {cached_data}")
            # Continue to reload data
    
    # Otherwise, load it now using our optimized approach
    with st.spinner(f"Loading {league} data..."):
        try:
            if league not in leagues:
                return f"League '{league}' not found in configured leagues"
                
            print(f"Attempting to fetch fresh data for {league}")
            sheet_name = leagues[league]
            
            # Use our optimized approach to get all tabs at once
            all_tabs_data = fetch_all_sheet_tabs(sheet_name)
            
            # If data is a string, it's an error message
            if isinstance(all_tabs_data, str):
                print(f"Error fetching data for {league}: {all_tabs_data}")
                return all_tabs_data
            
            # Process team data
            if "Power Rankings" in all_tabs_data:
                team_raw_data = all_tabs_data["Power Rankings"]
                processed_data = process_team_data(team_raw_data, sheet_name)
                
                # Store in session state for future use only if it's a valid DataFrame
                if not isinstance(processed_data, str):
                    st.session_state.all_data["team_data"][league] = processed_data
                    
                    # Also store all player data while we're at it
                    st.session_state.all_data["player_data"][league] = {}
                    for role in roles:
                        if role in all_tabs_data:
                            player_raw_data = all_tabs_data[role]
                            st.session_state.all_data["player_data"][league][role] = process_player_data(player_raw_data, role)
                
                return processed_data
            else:
                error_msg = "Power Rankings tab not found"
                print(f"Error for {league}: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error fetching {league} data: {str(e)}"
            print(error_msg)
            return error_msg

def get_player_data_with_fallback(league, role):
    """Get player data with fallback for when it's not in cache"""
    # Check if we already have this data
    if league in st.session_state.all_data["player_data"] and role in st.session_state.all_data["player_data"][league]:
        cached_data = st.session_state.all_data["player_data"][league][role]
        # Make sure the cached data is actually valid and not an error message
        if not isinstance(cached_data, str):
            return cached_data
        else:
            print(f"Cached player data for {league}/{role} is an error message: {cached_data}")
            # Continue to reload data
    
    # Make sure the league dict exists
    if league not in st.session_state.all_data["player_data"]:
        st.session_state.all_data["player_data"][league] = {}
    
    # Load the data now using our optimized approach
    with st.spinner(f"Loading {league} data..."):
        try:
            if league not in leagues:
                return f"League '{league}' not found in configured leagues"
                
            print(f"Attempting to fetch fresh data for {league}")
            sheet_name = leagues[league]
            
            # Use our optimized approach to get all tabs at once
            all_tabs_data = fetch_all_sheet_tabs(sheet_name)
            
            # If data is a string, it's an error message
            if isinstance(all_tabs_data, str):
                print(f"Error fetching data for {league}: {all_tabs_data}")
                return all_tabs_data
            
            # First process and store team data while we're at it
            if "Power Rankings" in all_tabs_data and league not in st.session_state.all_data["team_data"]:
                team_raw_data = all_tabs_data["Power Rankings"]
                st.session_state.all_data["team_data"][league] = process_team_data(team_raw_data, sheet_name)
            
            # Now process the player data for the requested role
            if role in all_tabs_data:
                player_raw_data = all_tabs_data[role]
                processed_data = process_player_data(player_raw_data, role)
                
                # Store in session state for future use
                st.session_state.all_data["player_data"][league][role] = processed_data
                return processed_data
            else:
                error_msg = f"{role} tab not found"
                print(f"Error for {league}: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error fetching {league}/{role} data: {str(e)}"
            print(error_msg)
            return error_msg

# Get team players function
@st.cache_data(ttl=3600)
def get_team_players(team_name, all_data):
    """Find all players from a specific team and organize them by role using official abbreviations"""
    team_players = {
        "Top": {"Player": "Unknown", "Score": 0},
        "Jungle": {"Player": "Unknown", "Score": 0},
        "Mid": {"Player": "Unknown", "Score": 0},
        "Bot": {"Player": "Unknown", "Score": 0},
        "Support": {"Player": "Unknown", "Score": 0}
    }
    
    # Official team abbreviations from regional scripts
    official_team_abbrs = {
        # LCP teams
        "Chiefs Esports Club": "CHF",
        "CTBC Flying Oyster": "CFO",
        "DetonatioN FocusMe": "DFM",
        "Fukuoka SoftBank HAWKS gaming": "SHG",
        "GAM Esports": "GAM",
        "MGN Vikings Esports": "MVKE",
        "TALON": "TLN",
        "Team Secret Whales": "TSW",
        
        # LEC teams
        "Fnatic": "FNC",
        "G2 Esports": "G2",
        "Team Vitality": "VIT",
        "SK Gaming": "SK",
        "Movistar KOI": "MDK",
        "Team BDS": "BDS",
        "GIANTX": "GX",
        "Karmine Corp": "KC",
        "Rogue": "RGE",
        "Team Heretics": "TH",
        
        # LTA North teams
        "100 Thieves": "100T",
        "Cloud9": "C9",
        "Dignitas": "DIG",
        "Disguised": "DSG",
        "FlyQuest": "FLY",
        "LYON": "LYON",
        "Shopify Rebellion": "SR",
        "Team Liquid": "TL",
        
        # LCK teams
        "BNK FearX": "BFX",
        "DN Freecs": "DNF",
        "Dplus KIA": "DK",
        "DRX": "DRX",
        "Gen.G eSports": "GEN",
        "Hanwha Life eSports": "HLE",
        "KT Rolster": "KT",
        "Nongshim RedForce": "NS",
        "OK BRION": "BRO",
        "T1": "T1",
        
        # LPL teams
        "Bilibili Gaming": "BLG",
        "Weibo Gaming": "WBG",
        "JD Gaming": "JDG",
        "Edward Gaming": "EDG",
        "LGD Gaming": "LGD",
        "Anyone s Legend": "AL",
        "TT": "TT",
        "Ninjas in Pyjamas": "NIP",
        "Top Esports": "TES",
        "Royal Never Give Up": "RNG",
        "Invictus Gaming": "IG",
        "OMG": "OMG",
        "Funplus Phoenix": "FPX",
        "LNG Esports": "LNG",
        "Ultra Prime": "UP",
        "Team WE": "WE",
    }
    
    # Get the official abbreviation for this team
    team_abbr = None
    # First try exact match
    if team_name in official_team_abbrs:
        team_abbr = official_team_abbrs[team_name]
    else:
        # Try partial matches
        for official_name, abbr in official_team_abbrs.items():
            # Check if the official name is contained in the selected team name or vice versa
            if official_name.lower() in team_name.lower() or team_name.lower() in official_name.lower():
                team_abbr = abbr
                break
    
    # If we didn't find an abbreviation, create one from the team name
    if not team_abbr:
        # Get first word or first character of each word
        words = team_name.split()
        if len(words) == 1:
            team_abbr = words[0][:3].upper()  # Use first 3 chars of single word
        else:
            team_abbr = ''.join(word[0] for word in words if word).upper()  # First letter of each word
    
    print(f"Using abbreviation '{team_abbr}' for team '{team_name}'")
    
    # Prepare a list of possible team prefix patterns
    team_patterns = [
        f"{team_abbr} ",       # Standard format: "TSM Bjergsen"
        f"{team_abbr}.",       # Format with dot: "TSM.Bjergsen"
        f"{team_abbr.lower()} ", # Lowercase: "tsm Bjergsen"
        f"{team_name} ",       # Full name: "Team SoloMid Bjergsen"
    ]
    
    # Also add commonly paired abbreviations
    common_paired_abbrs = {
        "T1": ["SKT", "SKT T1"],
        "GEN": ["GENG", "Gen.G"],
        "DK": ["DWG", "DWGKIA", "Damwon"],
        "KC": ["Karmine", "KarmineCorp"],
        "C9": ["Cloud9"],
        "DRX": ["DragonX"],
        "TL": ["Liquid"],
        "FNC": ["Fnatic"],
        "G2": ["G2Esports"],
        "RNG": ["Royal"],
    }
    
    if team_abbr in common_paired_abbrs:
        for alt_abbr in common_paired_abbrs[team_abbr]:
            team_patterns.extend([f"{alt_abbr} ", f"{alt_abbr}."])
    
    # Find all players from this team across all roles
    players_found = []
    
    # Search for players with team prefix
    for league_name in all_data["player_data"]:
        for role in roles:
            player_data = all_data["player_data"][league_name][role]
            if isinstance(player_data, str) or player_data.empty:
                continue
            
            # Check each player in this role
            for _, row in player_data.iterrows():
                player_name = row["Player"] if "Player" in row else ""
                if not isinstance(player_name, str) or not player_name:
                    continue
                
                # Try each pattern
                found_match = False
                for pattern in team_patterns:
                    if player_name.startswith(pattern) or player_name.lower().startswith(pattern.lower()):
                        found_match = True
                        break
                
                # Also check for player names that contain the abbreviation within the first few characters
                # Common in some regions: "BLGBin" instead of "BLG Bin"
                if not found_match and len(team_abbr) >= 2:
                    if team_abbr in player_name[:len(team_abbr)+2] or team_abbr.lower() in player_name.lower()[:len(team_abbr)+2]:
                        found_match = True
                
                if found_match:
                    player_info = {
                        "Player": player_name,
                        "Score": row["Score"],
                        "League": league_name,
                        "Role": role
                    }
                    players_found.append(player_info)
    
    # Now that we've found all players, assign them to the best matching role
    print(f"Found {len(players_found)} players for team {team_name} using abbreviation {team_abbr}: {[p['Player'] for p in players_found]}")
    
    # First, try to fill roles with exact role matches
    for player_info in players_found:
        role = player_info["Role"]
        if team_players[role]["Player"] == "Unknown":
            team_players[role] = player_info
    
    # If we have leftover players, try to assign them to any remaining empty roles
    remaining_players = [p for p in players_found if p["Player"] != team_players[p["Role"]]["Player"]]
    for role in roles:
        if team_players[role]["Player"] == "Unknown" and remaining_players:
            team_players[role] = remaining_players.pop(0)
    
    return team_players

# Add a helper function to display player names with team abbreviations for clarity
def format_player_name(player_name, team_name=None):
    """Format player names to clearly show their team association"""
    if not player_name or player_name == "Unknown":
        return player_name
        
    # Return as is if we don't have a team name
    if not team_name:
        return player_name
    
    # Check if the player name already starts with team name or abbreviation
    if any(player_name.lower().startswith(word.lower()) for word in team_name.split()):
        # Extract just the player name without the team prefix
        for word in team_name.split():
            if player_name.lower().startswith(word.lower()):
                # Find where the team name ends and extract just the player part
                idx = player_name.lower().find(word.lower()) + len(word)
                # Skip any separators like ":" or spaces
                while idx < len(player_name) and (player_name[idx] == ' ' or player_name[idx] == ':'):
                    idx += 1
                return player_name[idx:].strip()
        return player_name
    
    # Otherwise, just return the player name without team prefix
    return player_name

# Function to simulate matchups
def simulate_matchup(team_a_score, team_b_score, team_power_scores, simulations=10000, batch_size=1000, player_data=None):
    """Simulate a matchup using team power scores with logarithmic scaling and player data integration."""
    
    team_a_wins = 0
    team_b_wins = 0
    num_batches = simulations // batch_size

    # Calculate relative strength using log transformation
    # This compresses the 50-3000 range while preserving meaningful differences
    log_team_a = np.log(max(50, team_a_score))
    log_team_b = np.log(max(50, team_b_score))
    
    # Compute normalized score difference (as percentage of max possible difference)
    max_score_diff = np.log(3000) - np.log(50)
    normalized_diff = abs(log_team_a - log_team_b) / max_score_diff
    
    # Define variance based on normalized difference
    # Higher differences = lower variance (more predictable)
    base_variance = 0.25
    if normalized_diff <= 0.15:  # Small relative difference
        variance_factor = base_variance * 1.5  # Higher variance
    elif normalized_diff <= 0.3:  # Medium relative difference
        variance_factor = base_variance * 1.0  # Medium variance
    else:  # Large relative difference
        variance_factor = base_variance * 0.5  # Lower variance
    
    # Incorporate player data if available
    player_team_a_bonus = 0
    player_team_b_bonus = 0
    player_variance_adjustment = 0
    
    if player_data and 'a_weighted' in player_data and 'b_weighted' in player_data:
        # Convert player scores to log scale too
        a_weighted = player_data['a_weighted']
        b_weighted = player_data['b_weighted']
        
        # Calculate player impact (15-25% of total outcome)
        player_impact = 0.20  # Player data accounts for 20% of prediction
        
        # Calculate log player scores
        log_player_a = np.log(max(50, a_weighted * 10))  # Scale to similar range as team scores
        log_player_b = np.log(max(50, b_weighted * 10))
        
        # Add player bonus
        player_team_a_bonus = (log_player_a - log_player_b) * player_impact
        player_team_b_bonus = -player_team_a_bonus
        
        # Adjust variance based on player differences
        player_diff_ratio = abs(a_weighted - b_weighted) / max(a_weighted, b_weighted)
        player_variance_adjustment = -0.05 * player_diff_ratio  # Reduce variance when player gap is large
    
    # Adjust variance with player data
    variance_factor = max(0.1, variance_factor + player_variance_adjustment)

    print(f"Running {simulations} Simulations with Normalized Diff: {normalized_diff:.3f}, Variance Factor: {variance_factor:.3f}")
    if player_data:
        print(f"Player Bonuses: Team A: {player_team_a_bonus:.3f}, Team B: {player_team_b_bonus:.3f}")

    # Run simulations using log-transformed scores with player bonuses
    for _ in range(num_batches):
        # Variance is now based on position in the log scale, not raw score
        team_a_performance = np.random.normal(log_team_a + player_team_a_bonus, variance_factor, batch_size)
        team_b_performance = np.random.normal(log_team_b + player_team_b_bonus, variance_factor, batch_size)
        
        team_a_wins += np.sum(team_a_performance > team_b_performance)
        team_b_wins += np.sum(team_b_performance > team_a_performance)

    team_a_win_prob = round((team_a_wins / simulations) * 100, 1)
    team_b_win_prob = round((team_b_wins / simulations) * 100, 1)

    # Allow more extreme probabilities for large differences (5-95% range)
    team_a_win_prob = max(5, min(95, team_a_win_prob))
    team_b_win_prob = 100 - team_a_win_prob

    raw_score_diff = abs(team_a_score - team_b_score)
    print(f"Final Win Probability: {team_a_win_prob}% vs. {team_b_win_prob}% (Raw Score Diff: {raw_score_diff:.2f}, Log Diff: {abs(log_team_a - log_team_b):.3f})")

    return team_a_win_prob, team_b_win_prob

# Modify the start-up data loading code
if 'data_loaded' not in st.session_state:
    # Initialize last API call time
    st.session_state['last_api_call_time'] = time.time() - 5  # Start with a 5-second buffer
    
    # Try to load from the disk cache first before hitting the API
    all_data = safely_load_data()
    st.session_state.all_data = all_data
    st.session_state.data_loaded = True

# Streamlit UI
st.title("League of Legends Power Score Dashboard")

# Initialization info
if "all_data" in st.session_state:
    st.success("✅ Data successfully cached! App will now run without excessive API calls.")

st.sidebar.header("Select View Mode")
view_mode = st.sidebar.radio("Choose a View:", [
    "League-Based Player Rankings",
    "Global Player Rankings",
    "League-Based Team Rankings",
    "Global Team Rankings",
    "Match Prediction"
])

if view_mode == "League-Based Player Rankings":
    selected_league = st.sidebar.selectbox("Choose a League", list(leagues.keys()))
    selected_role = st.sidebar.selectbox("Choose a Role", roles)

    # Use cached data with fallback
    df = get_player_data_with_fallback(selected_league, selected_role)

    if isinstance(df, str):
        st.error(df)
    elif not df.empty:
        # Reset index to start from 1
        df = df.copy()  # Avoid modifying cached data
        df.index = range(1, len(df) + 1)

        st.write(f"### {selected_league} - {selected_role} Rankings")
        st.dataframe(df)
    else:
        st.warning("No data found. Check the sheet name or API connection.")

elif view_mode == "Global Player Rankings":
    selected_role = st.sidebar.selectbox("Choose a Role", roles)
    
    # Add a loading spinner like in Global Team Rankings
    st.write(f"### Global {selected_role} Rankings")
    
    with st.spinner(f"Fetching {selected_role} player data from all leagues (this may take a moment)..."):
        combined_df = pd.DataFrame()
        loaded_leagues = []
        failed_leagues = []

        # Combine player data from all leagues using fallback function for better error handling
        for league_name, sheet_name in leagues.items():
            try:
                # Use the fallback function to get player data with retries
                temp_df = get_player_data_with_fallback(league_name, selected_role)
                
                if isinstance(temp_df, str):
                    # This is an error message
                    failed_leagues.append(f"{league_name}: {temp_df}")
                    continue
                    
                if not temp_df.empty and "Player" in temp_df.columns and "Score" in temp_df.columns:
                    # Create a copy to avoid modifying cached data
                    temp_df = temp_df.copy()
                    temp_df["League"] = league_name  # Ensure each row keeps its region
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
                    loaded_leagues.append(league_name)
                else:
                    failed_leagues.append(f"{league_name}: Missing required columns")
            except Exception as e:
                failed_leagues.append(f"{league_name}: {str(e)}")
                # Continue with other leagues even if one fails
                continue
    
    # Display any loading errors but don't block viewing available data
    if failed_leagues:
        st.warning(f"Some leagues couldn't be loaded: {', '.join(failed_leagues)}")
    
    if loaded_leagues:
        st.success(f"Successfully loaded data from: {', '.join(loaded_leagues)}")

    if not combined_df.empty:
        # Ensure Score column is numeric for proper sorting
        combined_df.loc[:, "Score"] = pd.to_numeric(combined_df["Score"], errors="coerce")
        
        # Fill any NaN values in Score with a default (0)
        combined_df["Score"] = combined_df["Score"].fillna(0).infer_objects(copy=False)

        # Sort by Power Score (Descending)
        combined_df = combined_df.sort_values(by="Score", ascending=False)

        # Reset index to start from 1
        combined_df.index = range(1, len(combined_df) + 1)
        
        # Add some metrics about the data
        st.write(f"Displaying {len(combined_df)} players from {len(loaded_leagues)} leagues.")

        # Display the dataframe with enhanced styling
        st.dataframe(combined_df, use_container_width=True)
    else:
        st.error("No player data could be loaded. This could be due to API rate limits or connection issues.")
        st.info("Try these steps to resolve the issue:")
        st.info("1. Wait a few minutes and try again")
        st.info("2. Try the 'Refresh All Data' button in the sidebar")
        st.info("3. Check individual league player rankings to see if they load correctly")

elif view_mode == "League-Based Team Rankings":
    selected_league = st.sidebar.selectbox("Choose a League", list(leagues.keys()))

    # Use cached team data with fallback
    df = get_team_data_with_fallback(selected_league)

    if isinstance(df, str):
        st.error(df)
    elif not df.empty:
        # Create a copy to avoid modifying cached data
        df = df.copy()
        
        # Reset index to start from 1
        df.index = range(1, len(df) + 1)

        st.write(f"### {selected_league} - Team Rankings")
        st.dataframe(df)
    else:
        st.warning("No data found. Check the sheet name or API connection.")

elif view_mode == "Global Team Rankings":
    st.write("### Global Team Rankings")
    
    # Add a loading spinner to indicate data is being fetched
    with st.spinner("Fetching team data from all leagues (this may take a moment)..."):
        combined_df = pd.DataFrame()
        loaded_leagues = []
        failed_leagues = []

        # Combine team data from all leagues with better error handling
        for league_name, sheet_name in leagues.items():
            try:
                # Use the fallback function to get team data with retries
                temp_df = get_team_data_with_fallback(league_name)
                
                if isinstance(temp_df, str):
                    # This is an error message
                    failed_leagues.append(f"{league_name}: {temp_df}")
                    continue
                    
                if not temp_df.empty and "Team" in temp_df.columns and "Score" in temp_df.columns:
                    # Create a copy to avoid modifying cached data
                    temp_df = temp_df.copy()
                    temp_df["League"] = league_name
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
                    loaded_leagues.append(league_name)
                else:
                    failed_leagues.append(f"{league_name}: Missing required columns")
            except Exception as e:
                failed_leagues.append(f"{league_name}: {str(e)}")
                # Continue with other leagues even if one fails
                continue
    
    # Display any loading errors but don't block viewing available data
    if failed_leagues:
        st.warning(f"Some leagues couldn't be loaded: {', '.join(failed_leagues)}")
    
    if loaded_leagues:
        st.success(f"Successfully loaded data from: {', '.join(loaded_leagues)}")

    if not combined_df.empty:
        # Ensure Score column is numeric for proper sorting
        combined_df.loc[:, "Score"] = pd.to_numeric(combined_df["Score"], errors="coerce")
        
        # Fill any NaN values in Score with a default (0)
        combined_df["Score"] = combined_df["Score"].fillna(0).infer_objects(copy=False)

        # Sort by Power Score (Descending)
        combined_df = combined_df.sort_values(by="Score", ascending=False)

        # Reset index to start from 1
        combined_df.index = range(1, len(combined_df) + 1)

        # Add some metrics about the data
        st.write(f"Displaying {len(combined_df)} teams from {len(loaded_leagues)} leagues.")
        
        # Display the dataframe with enhanced styling
        st.dataframe(combined_df, use_container_width=True)
    else:
        st.error("No team data could be loaded. This could be due to API rate limits or connection issues.")
        st.info("Try these steps to resolve the issue:")
        st.info("1. Wait a few minutes and try again")
        st.info("2. Try the 'Refresh All Data' button in the sidebar")
        st.info("3. Check individual league team rankings to see if they load correctly")

elif view_mode == "Match Prediction":
    # Add debugging options
    debug_mode = st.sidebar.checkbox("Enable debugging mode", value=False)
    
    if debug_mode:
        st.write("### Debug Information")
        
        # Add a detailed API test function
        if st.button("Detailed API Test"):
            with st.spinner("Running detailed API test..."):
                try:
                    test_results = []
                    st.write("Testing connection to Google Sheets API...")
                    
                    # Test basic connectivity
                    st.write("### 1. Basic Connectivity Test")
                    try:
                        # List spreadsheets to test basic connectivity
                        sheets = client.list_spreadsheet_files()
                        accessible_sheets = [sheet['name'] for sheet in sheets]
                        st.success(f"✅ Successfully connected to Google Sheets API. Found {len(accessible_sheets)} accessible spreadsheets.")
                        st.write("First 5 accessible spreadsheets:")
                        for i, name in enumerate(accessible_sheets[:5]):
                            st.write(f"- {name}")
                    except Exception as e:
                        st.error(f"❌ Basic connectivity test failed: {str(e)}")
                    
                    # Test each league with both methods
                    st.write("### 2. League Data Retrieval Test")
                    for league_name, sheet_name in leagues.items():
                        st.write(f"#### Testing {league_name} ({sheet_name})")
                        
                        # Test method 1: Standard worksheet approach
                        try:
                            st.write("Method 1: Standard worksheet approach")
                            sheet = client.open(sheet_name).worksheet("Power Rankings")
                            data = sheet.get_all_values()
                            
                            if hasattr(data, 'status_code'):
                                st.warning(f"⚠️ Got Response object with status code {data.status_code}")
                                st.write(f"Response type: {type(data)}")
                                try:
                                    st.write(f"Response content sample: {str(data.content)[:200]}")
                                except:
                                    st.write("Could not access response content")
                            elif isinstance(data, list):
                                st.success(f"✅ Successfully retrieved {len(data)} rows using method 1")
                                if len(data) > 0:
                                    st.write("Sample data (first row):")
                                    st.write(data[0])
                            else:
                                st.warning(f"⚠️ Unexpected data type: {type(data)}")
                        except Exception as e:
                            st.error(f"❌ Method 1 failed: {str(e)}")
                        
                        # Test method 2: Direct values_get approach
                        try:
                            st.write("Method 2: Direct values_get approach")
                            spreadsheet = client.open(sheet_name)
                            spreadsheet_id = spreadsheet.id
                            result = client.values_get(
                                spreadsheetId=spreadsheet_id,
                                range="'Power Rankings'!A1:Z1000"
                            )
                            direct_data = result.get('values', [])
                            st.success(f"✅ Successfully retrieved {len(direct_data)} rows using method 2")
                            if len(direct_data) > 0:
                                st.write("Sample data (first row):")
                                st.write(direct_data[0])
                        except Exception as e:
                            st.error(f"❌ Method 2 failed: {str(e)}")
                    
                    # Test the cached data
                    st.write("### 3. Cached Data Test")
                    for league, data in st.session_state.all_data["team_data"].items():
                        if isinstance(data, str):
                            st.warning(f"{league}: {data}")
                        elif data is None:
                            st.warning(f"{league}: None")
                        else:
                            st.success(f"{league}: DataFrame with {len(data)} rows")
                            st.write(f"Columns: {data.columns.tolist()}")
                            
                except Exception as e:
                    st.error(f"❌ API test failed with exception: {str(e)}")
        
        if st.button("Test API Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    # Test a simple API call
                    test_league = list(leagues.keys())[0]
                    test_sheet = leagues[test_league]
                    test_data = fetch_raw_sheet_data(test_sheet, "Power Rankings")
                    
                    if isinstance(test_data, str):
                        st.error(f"API connection test failed: {test_data}")
                    else:
                        st.success(f"API connection successful! Retrieved {len(test_data)} rows of data.")
                        st.write("First 3 rows of raw data:")
                        for i in range(min(3, len(test_data))):
                            st.write(test_data[i])
                except Exception as e:
                    st.error(f"API connection test failed with exception: {str(e)}")
        
        if st.button("View Team Data Cache"):
            st.write("Current team data in cache:")
            for league, data in st.session_state.all_data["team_data"].items():
                if isinstance(data, str):
                    st.warning(f"{league}: {data}")
                elif data is None:
                    st.warning(f"{league}: None")
                else:
                    st.success(f"{league}: DataFrame with {len(data)} rows")
        
        if st.button("Clear and Reload All Data"):
            # Clear session state variables
            st.session_state.pop('data_loaded', None)
            st.session_state.pop('all_data', None)
            st.session_state.pop('global_team_df', None)
            st.session_state.pop('loaded_leagues', None)
            st.session_state.pop('failed_leagues', None)
            st.session_state.pop('all_players_by_role', None)
            
            # Clear all cache data functions explicitly
            fetch_raw_sheet_data.clear()
            fetch_all_sheet_tabs.clear()
            get_player_data.clear()
            get_team_data.clear()
            load_all_data.clear()
            load_all_data_with_workers.clear()
            get_team_players.clear()
            
            # Also clear any disk cache
            try:
                cache_dir = ".sheet_cache"
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                st.warning(f"Could not clear disk cache: {e}")
            
            # Force a complete rerun
            st.rerun()
    
    # Create global team dataframe from cached data
    st.write("### Match Prediction")
    with st.spinner("Loading team data for match prediction..."):
        # First check if we have global team data in session state
        if 'global_team_df' not in st.session_state or st.session_state.get('global_team_df', None) is None:
            # Build the global team dataframe the same way as Global Team Rankings tab
            global_team_df = pd.DataFrame()
            loaded_leagues = []
            failed_leagues = []

            # Combine team data from all leagues with robust error handling
            for league_name, sheet_name in leagues.items():
                try:
                    # Use the fallback function to get team data with retries
                    temp_df = get_team_data_with_fallback(league_name)
                    
                    if isinstance(temp_df, str):
                        failed_leagues.append(f"{league_name}: {temp_df}")
                        continue
                        
                    if not temp_df.empty and "Team" in temp_df.columns and "Score" in temp_df.columns:
                        # Create a copy to avoid modifying cached data
                        temp_df = temp_df.copy()
                        temp_df["League"] = league_name
                        global_team_df = pd.concat([global_team_df, temp_df], ignore_index=True)
                        loaded_leagues.append(league_name)
                    else:
                        failed_leagues.append(f"{league_name}: Missing required columns")
                except Exception as e:
                    failed_leagues.append(f"{league_name}: {str(e)}")
                    continue
                    
            # Save to session state for future use
            st.session_state['global_team_df'] = global_team_df
            st.session_state['loaded_leagues'] = loaded_leagues
            st.session_state['failed_leagues'] = failed_leagues
        else:
            # Use the already loaded data
            global_team_df = st.session_state['global_team_df']
            loaded_leagues = st.session_state.get('loaded_leagues', [])
            failed_leagues = st.session_state.get('failed_leagues', [])
            
        # Display any loading errors
        if failed_leagues:
            st.warning(f"Some leagues couldn't be loaded: {', '.join(failed_leagues)}")
            
        if loaded_leagues:
            st.success(f"Successfully loaded team data from: {', '.join(loaded_leagues)}")
    
    if global_team_df.empty:
        st.error("Could not load any team data. Please check your Google Sheets connection.")
        st.info("Try clicking the 'Refresh All Data' button in the sidebar to reset the cache and try again.")
        
        # Add a direct reload button
        if st.button("Reload Team Data Now"):
            # Clear only the team data
            if 'global_team_df' in st.session_state:
                del st.session_state['global_team_df']
            st.rerun()
    else:
        # Ensure Score is treated as numeric
        if "Score" in global_team_df.columns:
            global_team_df.loc[:, "Score"] = pd.to_numeric(global_team_df["Score"], errors="coerce")
            global_team_df = global_team_df.sort_values(by="Score", ascending=False)

            # Get unique teams for dropdowns
            teams = sorted(global_team_df["Team"].unique())

            # Team selection dropdowns
            col1, col2 = st.columns(2)
            with col1:
                team_a = st.selectbox("Select Team A", teams)
            with col2:
                team_b = st.selectbox("Select Team B", teams)

            if team_a and team_b:
                st.write(f"### Match Prediction: {team_a} vs. {team_b}")

                try:
                    # Retrieve team power scores from Global Team Rankings
                    team_a_data = global_team_df.loc[global_team_df["Team"] == team_a]
                    team_b_data = global_team_df.loc[global_team_df["Team"] == team_b]
                    
                    if team_a_data.empty or team_b_data.empty:
                        st.error(f"Could not find data for {'Team A' if team_a_data.empty else 'Team B'}")
                    else:
                        team_a_score = team_a_data["Score"].values[0]
                        team_b_score = team_b_data["Score"].values[0]

                        # Setup for matchup - calculate all metrics first before passing to simulator
                        try:
                            # Create team player matchups first
                            team_a_players = get_team_players(team_a, st.session_state.all_data)
                            team_b_players = get_team_players(team_b, st.session_state.all_data)
                            
                            # Prepare for manual player selection fallback
                            if 'all_players_by_role' not in st.session_state:
                                st.session_state['all_players_by_role'] = {}
                            
                            all_players_by_role = {}
                            for role in roles:
                                # Check if we already have this data in session state
                                if role in st.session_state['all_players_by_role']:
                                    all_players_by_role[role] = st.session_state['all_players_by_role'][role]
                                    continue
                                
                                # Otherwise, load the data
                                all_players_by_role[role] = []
                                
                                # Show loading indicator
                                with st.spinner(f"Loading {role} player data (one-time operation)..."):
                                    # Collect all players for this role across leagues
                                    for league in leagues:
                                        try:
                                            player_data = get_player_data_with_fallback(league, role)
                                            if not isinstance(player_data, str) and not player_data.empty and "Player" in player_data.columns:
                                                for _, row in player_data.iterrows():
                                                    player_name = row["Player"]
                                                    player_score = row["Score"]
                                                    if pd.notna(player_name) and pd.notna(player_score):
                                                        all_players_by_role[role].append({
                                                            "name": player_name,
                                                            "score": player_score,
                                                            "league": league
                                                        })
                                        except Exception as e:
                                            print(f"Error loading {league} {role} player data: {str(e)}")
                                
                                # Cache the results for future use
                                st.session_state['all_players_by_role'][role] = all_players_by_role[role]
                            
                            # Define role weights (some roles have more impact on the game)
                            role_weights = {
                                "Top": 0.17,
                                "Jungle": 0.25,
                                "Mid": 0.23,
                                "Bot": 0.22,
                                "Support": 0.13
                            }
                            
                            # Calculate individual role advantages
                            matchup_data = []
                            for role in roles:
                                a_player = team_a_players[role]["Player"]
                                a_score = team_a_players[role]["Score"]
                                b_player = team_b_players[role]["Player"]
                                b_score = team_b_players[role]["Score"]
                                
                                diff = a_score - b_score
                                if abs(diff) < 50:
                                    advantage = "Even"
                                else:
                                    advantage = f"{a_player if diff > 0 else b_player} (+{abs(round(diff, 1))})"
                                
                                # Add player comparison row
                                matchup_data.append({
                                    "Role": role,
                                    "Team A": a_player,
                                    "Score A": round(a_score, 1),
                                    "Team B": b_player,
                                    "Score B": round(b_score, 1),
                                    "Advantage": advantage
                                })
                            
                            # Calculate player totals and advantages
                            a_total = sum(team_a_players[role]["Score"] for role in roles)
                            b_total = sum(team_b_players[role]["Score"] for role in roles)
                            a_weighted_total = sum(team_a_players[role]["Score"] * role_weights[role] for role in roles)
                            b_weighted_total = sum(team_b_players[role]["Score"] * role_weights[role] for role in roles)
                            a_advantages = sum(1 for match in matchup_data if match["Advantage"] != "Even" and match["Team A"] in match["Advantage"])
                            b_advantages = sum(1 for match in matchup_data if match["Advantage"] != "Even" and match["Team B"] in match["Advantage"])
                            
                            # Pass player data to matchup simulator
                            player_data = {
                                'a_weighted': a_weighted_total,
                                'b_weighted': b_weighted_total,
                                'a_advantages': a_advantages,
                                'b_advantages': b_advantages
                            }
                            
                            # Simulate matchup with complete player data
                            team_a_win_prob, team_b_win_prob = simulate_matchup(team_a_score, team_b_score, {}, player_data=player_data)
                            
                            # Define predicted winner after simulation
                            predicted_winner = team_a if team_a_win_prob > team_b_win_prob else team_b
                            predicted_loser = team_b if predicted_winner == team_a else team_a
                            
                            # Safely extract values for kill prediction
                            team_a_kpg = team_a_data["Kills/Game"].iloc[0] if "Kills/Game" in team_a_data.columns else 10
                            team_b_kpg = team_b_data["Kills/Game"].iloc[0] if "Kills/Game" in team_b_data.columns else 10
                            team_a_dpg = team_a_data["Deaths/Game"].iloc[0] if "Deaths/Game" in team_a_data.columns else 10
                            team_b_dpg = team_b_data["Deaths/Game"].iloc[0] if "Deaths/Game" in team_b_data.columns else 10

                            # Ensure values are numeric, default to 10 if missing
                            team_a_kpg = float(team_a_kpg) if pd.notna(team_a_kpg) else 10
                            team_b_kpg = float(team_b_kpg) if pd.notna(team_b_kpg) else 10
                            team_a_dpg = float(team_a_dpg) if pd.notna(team_a_dpg) else 10
                            team_b_dpg = float(team_b_dpg) if pd.notna(team_b_dpg) else 10

                            # Predict expected kills and kill spread - using direct statistical approach
                            
                            # First, collect all the league-wide data we'll need throughout the prediction
                            # Initialize collection variables
                            league_kpg_values = []
                            league_game_times = []
                            league_kpm_values = []
                            has_game_time_data = False
                            team_a_time = 0
                            team_b_time = 0
                            team_a_kpm = 0
                            team_b_kpm = 0
                            
                            # Collect data from the global team dataframe
                            for _, team_row in global_team_df.iterrows():
                                # Collect KPG values
                                if "Kills/Game" in team_row and pd.notna(team_row["Kills/Game"]):
                                    try:
                                        kpg = float(team_row["Kills/Game"])
                                        if kpg > 0:  # Ensure valid value
                                            league_kpg_values.append(kpg)
                                    except (ValueError, TypeError):
                                        continue
                                
                                # Collect game time data
                                if "Avg Game Time" in team_row and pd.notna(team_row["Avg Game Time"]):
                                    try:
                                        game_time = float(team_row["Avg Game Time"])
                                        if game_time > 0:  # Ensure valid value
                                            league_game_times.append(game_time)
                                            
                                            # Check if this is one of our teams
                                            if team_row["Team"] == team_a:
                                                team_a_time = game_time
                                            elif team_row["Team"] == team_b:
                                                team_b_time = game_time
                                            
                                            # If we have KPG data too, calculate KPM
                                            if "Kills/Game" in team_row and pd.notna(team_row["Kills/Game"]):
                                                try:
                                                    kpg = float(team_row["Kills/Game"])
                                                    if kpg > 0 and game_time > 0:
                                                        kpm = kpg / (game_time / 60)  # Convert to minutes
                                                        league_kpm_values.append(kpm)
                                                        
                                                        # Store team KPM if this is one of our teams
                                                        if team_row["Team"] == team_a:
                                                            team_a_kpm = kpm
                                                        elif team_row["Team"] == team_b:
                                                            team_b_kpm = kpm
                                                except:
                                                    continue
                                    except (ValueError, TypeError):
                                        continue
                            
                            # Check if we have enough game time data to use
                            has_game_time_data = team_a_time > 0 and team_b_time > 0 and len(league_game_times) > 0
                            
                            # 1. Calculate the true statistical expectation based on team tendencies
                            # The deaths a team deals is essentially the kills they create
                            team_a_creates = team_a_kpg   # A's offensive contribution
                            team_b_creates = team_b_kpg   # B's offensive contribution
                            team_a_allows = team_a_dpg    # A's defensive weakness
                            team_b_allows = team_b_dpg    # B's defensive weakness
                            
                            # Calculate expected kills directly without arbitrary multipliers
                            # IMPROVED OBJECTIVE APPROACH:
                            
                            # First, calculate expected game duration to use in KPM calculations
                            # This needs to be calculated before kill predictions
                            if has_game_time_data and team_a_time > 0 and team_b_time > 0:
                                # If we have specific game time data for both teams, use their averages
                                # Teams with longer average games tend to play longer games against each other
                                base_duration = (team_a_time + team_b_time) / 2
                                
                                # Adjust for team kill rates (higher combined KPM = faster games)
                                if team_a_kpm > 0 and team_b_kpm > 0:
                                    combined_kpm = (team_a_kpm + team_b_kpm) / 2
                                    
                                    # Calculate average KPM across the league if we have that data
                                    if league_kpm_values and len(league_kpm_values) > 0:
                                        avg_kpm = sum(league_kpm_values) / len(league_kpm_values)
                                        # Higher than average KPM = shorter games
                                        kpm_factor = avg_kpm / combined_kpm if combined_kpm > 0 else 1.0
                                        # Apply a moderate adjustment based on KPM (max ~10% change)
                                        base_duration *= max(0.9, min(1.1, kpm_factor))
                            elif league_game_times and len(league_game_times) > 0:
                                # If we don't have team-specific times, use league average
                                avg_game_time = sum(league_game_times) / len(league_game_times)
                                base_duration = avg_game_time
                            else:
                                # Fallback to a reasonable average if no data
                                base_duration = 32.0 * 60  # League of Legends average game time in SECONDS
                            
                            # Competitiveness factor - closer games tend to be longer
                            # Use win probability to estimate competitiveness
                            win_prob_diff = abs(team_a_win_prob - team_b_win_prob) / 100
                            
                            # Statistically, very one-sided games tend to be 5-15% shorter
                            if win_prob_diff > 0.3:  # Significant skill gap
                                competitiveness_factor = 1.0 - (win_prob_diff - 0.3) * 0.3  # Max 9% reduction
                            else:  # Close matchup
                                competitiveness_factor = 1.0 + (0.3 - win_prob_diff) * 0.1  # Max 3% increase
                            
                            # Apply competitiveness factor
                            expected_duration = base_duration * competitiveness_factor
                            
                            # Add minimal natural variation (±1 minute)
                            expected_duration += np.random.uniform(-60, 60)  # Convert to seconds (±1 minute)
                            
                            # Now calculate kills using a more objective approach
                            
                            # Calculate normalized KPM for each team
                            # If we don't have game time data, estimate using 30 min as default game length
                            team_a_kpm = team_a_kpm if team_a_kpm > 0 else team_a_kpg / (team_a_time / 60 if team_a_time > 0 else 30)
                            team_b_kpm = team_b_kpm if team_b_kpm > 0 else team_b_kpg / (team_b_time / 60 if team_b_time > 0 else 30)
                            
                            # Calculate deaths per minute
                            team_a_dpm = team_a_dpg / (team_a_time / 60 if team_a_time > 0 else 30)
                            team_b_dpm = team_b_dpg / (team_b_time / 60 if team_b_time > 0 else 30)

                            # Instead of using artificial minimum/maximum KPM values, let's use actual data
                            # and adjust our base calculation approach to get more realistic numbers
                            
                            # Store original values for later reference
                            raw_team_a_kpm = team_a_kpm 
                            raw_team_b_kpm = team_b_kpm
                            raw_team_a_dpm = team_a_dpm
                            raw_team_b_dpm = team_b_dpm
                            
                            # Get league average KPM (used for reference)
                            if league_kpm_values and len(league_kpm_values) > 0:
                                avg_kpm = sum(league_kpm_values) / len(league_kpm_values)
                            else:
                                # Fallback to reasonable average based on professional LoL data
                                avg_kpm = 0.85
                            
                            # Get league average DPM
                            avg_dpm = avg_kpm  # In aggregate, kills = deaths across the league
                            
                            # Calculate relative team strengths more directly from the raw data
                            # For professional LoL, we know:
                            # 1. Better teams get more kills and die less
                            # 2. Total kills in a game is a function of both teams' playstyles
                            # 3. Kill spread is a function of team skill difference
                            
                            # Calculate total expected kills using a function of both teams' kill tendencies
                            # This is a more direct statistical approach based on actual match data
                            
                            # Calculate expected game duration properly
                            # Expected duration is in seconds here
                            expected_duration_mins = expected_duration / 60  # Convert to minutes
                            
                            # Calculate kill tendencies (how many kills this team gets against average opposition)
                            team_a_kill_tendency = team_a_kpg
                            team_b_kill_tendency = team_b_kpg
                            
                            # Calculate death tendencies (how many deaths this team allows against average opposition)
                            team_a_death_tendency = team_a_dpg
                            team_b_death_tendency = team_b_dpg
                            
                            # FUNDAMENTAL CALCULATION: Better reflect how teams perform against each other
                            # When teams play, their offensive and defensive strengths interact
                            # BLG vs EDG: ~29-30 kills (16-20 for BLG, 10-13 for EDG)
                            # BLG vs WBG: ~24-26 kills with closer distribution
                            
                            # Calculate offensive strength as a ratio to league average
                            a_off_strength = team_a_kpg / np.mean(league_kpg_values) if league_kpg_values else team_a_kpg / 14.0
                            b_off_strength = team_b_kpg / np.mean(league_kpg_values) if league_kpg_values else team_b_kpg / 14.0
                            
                            # Calculate defensive strength as inverse ratio (lower deaths = better defense)
                            a_def_strength = (np.mean(league_kpg_values) if league_kpg_values else 14.0) / team_a_dpg if team_a_dpg > 0 else 1.0
                            b_def_strength = (np.mean(league_kpg_values) if league_kpg_values else 14.0) / team_b_dpg if team_b_dpg > 0 else 1.0
                            
                            # Calculate team skill ratio using power scores
                            team_skill_ratio = team_a_score / team_b_score if team_b_score > 0 else 1.0
                            
                            # Calculate expected kills for each team
                            # Team A kills = Their offensive strength vs B's defensive strength
                            # (adjusted by game duration and skill ratio)
                            
                            # Base expected kills (accounting for game duration)
                            # Normal LoL game has ~28-30 kills in ~32 minutes
                            # Scale based on expected duration
                            avg_game_length = 32.0 * 60  # 32 minutes in seconds
                            duration_factor = expected_duration / avg_game_length
                            
                            # ENHANCED, DATA-DRIVEN KILL PREDICTION APPROACH
                            # Using both KPG and DPG (deaths per game) for better accuracy
                            
                            # We already extracted team_a_dpg and team_b_dpg earlier, no need to redefine
                            # Ensure they're properly converted to float
                            try:
                                team_a_dpg = float(team_a_dpg)
                                team_b_dpg = float(team_b_dpg)
                            except (ValueError, TypeError):
                                # Fallback if conversion fails
                                team_a_dpg = float(team_a_kpg)
                                team_b_dpg = float(team_b_kpg)
                            
                            # Calculate team performance factors - how much their offense contributes to their games
                            team_a_offense_factor = team_a_kpg / (team_a_kpg + team_a_dpg) if (team_a_kpg + team_a_dpg) > 0 else 0.5
                            team_b_offense_factor = team_b_kpg / (team_b_kpg + team_b_dpg) if (team_b_kpg + team_b_dpg) > 0 else 0.5
                            
                            # More accurate prediction formula: combining offensive and defensive performance
                            # Team A's expected kills = their offense vs B's defense
                            # Team B's expected kills = their offense vs A's defense
                            team_a_expected_kills = ((team_a_kpg + team_b_dpg) / 2) * duration_factor
                            team_b_expected_kills = ((team_b_kpg + team_a_dpg) / 2) * duration_factor
                            
                            # Apply adjustments based on win probability
                            # Make sure win probability values are floats
                            try:
                                team_a_win_prob = float(team_a_win_prob)
                                team_b_win_prob = float(team_b_win_prob)
                            except (ValueError, TypeError):
                                # Default to 50-50 if conversion fails
                                team_a_win_prob = 50.0
                                team_b_win_prob = 50.0
                                
                            win_prob_factor = abs(team_a_win_prob - team_b_win_prob) / 100 * 0.2  # Max 20% adjustment
                            
                            if team_a_win_prob > team_b_win_prob:
                                team_a_expected_kills *= (1 + win_prob_factor)
                                team_b_expected_kills *= (1 - win_prob_factor)
                            else:
                                team_a_expected_kills *= (1 - win_prob_factor)
                                team_b_expected_kills *= (1 + win_prob_factor)
                            
                            # Calculate the expected total kills
                            expected_kills = team_a_expected_kills + team_b_expected_kills
                            
                            # Analyze league average kills for better limits
                            if league_kpg_values and len(league_kpg_values) > 0:
                                avg_league_kills = sum(league_kpg_values) * 2 / len(league_kpg_values)  # Approx. avg kills per match
                                min_realistic_kills = max(15.0, avg_league_kills * 0.7)  # At least 70% of league avg
                                max_realistic_kills = min(45.0, avg_league_kills * 1.4)  # At most 140% of league avg
                            else:
                                # Fallback if no league data
                                min_realistic_kills = 15.0
                                max_realistic_kills = 40.0
                            
                            # Apply data-driven limits for realism
                            expected_kills = max(min_realistic_kills, min(max_realistic_kills, expected_kills))
                            
                            # Keep track of the proportion each team contributes to total kills
                            # This is more sophisticated than just using KPG alone
                            team_a_kill_proportion = team_a_expected_kills / (team_a_expected_kills + team_b_expected_kills) if (team_a_expected_kills + team_b_expected_kills) > 0 else 0.5
                            
                            # For kill spread, use power scores, win probability and role advantages
                            # Base kill spread as a function of win probability
                            base_spread = (abs(team_a_win_prob - team_b_win_prob) / 15)  # More aggressive - up to 6.7 spread based on win probability
                            
                            # Safely ensure team scores are floats for calculations
                            try:
                                team_a_score = float(team_a_score)
                                team_b_score = float(team_b_score)
                            except (ValueError, TypeError):
                                # Default to equal scores if conversion fails
                                team_a_score = 1000.0
                                team_b_score = 1000.0
                            
                            # Add power score difference factor (0.6 kills per 100 power score difference)
                            power_score_diff = abs(team_a_score - team_b_score)
                            power_score_factor = (power_score_diff / 100) * 0.6
                            
                            # Factor in lane advantages more significantly
                            lane_advantage_kills = 0
                            for match in matchup_data:
                                if match["Advantage"] != "Even":
                                    # Add adjustment for each significant lane advantage
                                    score_diff = abs(match["Score A"] - match["Score B"])
                                    # Different roles have different impact on kill spread
                                    role_multiplier = 1.0
                                    if match["Role"] == "Jungle":
                                        role_multiplier = 1.2  # Jungle has more impact on kills
                                    elif match["Role"] == "Mid":
                                        role_multiplier = 1.1  # Mid has higher impact
                                    lane_advantage_kills += min(0.7, (score_diff / 400) * role_multiplier)
                            
                            # Calculate the total expected kill spread
                            expected_spread = base_spread + power_score_factor + lane_advantage_kills
                            expected_spread = max(1.5, min(16.0, expected_spread))  # Expanded but still reasonable bounds
                            
                            # Adjust expected kills for each team based on the spread
                            # Team with higher win probability gets more kills
                            if team_a_win_prob > team_b_win_prob:
                                # Start with proportional distribution based on our calculated kill proportion
                                raw_a_kills = expected_kills * team_a_kill_proportion
                                raw_b_kills = expected_kills * (1 - team_a_kill_proportion)
                                
                                # Current spread based on raw distribution
                                current_spread = raw_a_kills - raw_b_kills
                                
                                # Adjustment needed to reach expected spread
                                adjustment = (expected_spread - current_spread) / 2
                                
                                # Apply adjustment to reach the expected spread
                                a_expected_kills = raw_a_kills + adjustment
                                b_expected_kills = raw_b_kills - adjustment
                            else:
                                # Start with proportional distribution
                                raw_a_kills = expected_kills * team_a_kill_proportion
                                raw_b_kills = expected_kills * (1 - team_a_kill_proportion)
                                
                                # Current spread based on raw distribution
                                current_spread = raw_b_kills - raw_a_kills
                                
                                # Adjustment needed to reach expected spread
                                adjustment = (expected_spread - current_spread) / 2
                                
                                # Apply adjustment to reach the expected spread
                                a_expected_kills = raw_a_kills - adjustment
                                b_expected_kills = raw_b_kills + adjustment
                            
                            # Ensure no team gets fewer than 6 kills (minimum for professional games)
                            # This minimum is slightly higher than before based on real-world data
                            a_expected_kills = max(6.0, a_expected_kills)
                            b_expected_kills = max(6.0, b_expected_kills)
                            
                            # Recalculate total kills and kill spread
                            expected_kills = a_expected_kills + b_expected_kills
                            raw_kill_spread = abs(a_expected_kills - b_expected_kills)
                            
                            # Keep pace calculation for display purposes only
                            if league_kpg_values and len(league_kpg_values) > 0:
                                avg_kpg = sum(league_kpg_values) / len(league_kpg_values)
                                team_a_pace = team_a_kpg / avg_kpg if avg_kpg > 0 else 1.0
                                team_b_pace = team_b_kpg / avg_kpg if avg_kpg > 0 else 1.0
                                match_pace = (team_a_pace + team_b_pace) / 2
                            else:
                                team_a_pace = 1.0
                                team_b_pace = 1.0
                                match_pace = 1.0
                                
                            # Ensure kill spread direction matches winner
                            if (predicted_winner == team_a and a_expected_kills < b_expected_kills) or \
                               (predicted_winner == team_b and b_expected_kills < a_expected_kills):
                                # Swap kill expectations to match predicted winner
                                a_expected_kills, b_expected_kills = b_expected_kills, a_expected_kills
                                
                            # Calculate kill spread
                            raw_kill_spread = a_expected_kills - b_expected_kills if predicted_winner == team_a else b_expected_kills - a_expected_kills
                            kill_spread = abs(raw_kill_spread)
                            
                            # Round values for display
                            expected_kills = round(expected_kills, 1)
                            a_expected_kills = round(a_expected_kills, 1)
                            b_expected_kills = round(b_expected_kills, 1)
                            kill_spread = round(kill_spread, 1)

                            # Display match prediction results
                            st.write(f"🏆 **Predicted Winner:** {predicted_winner} ({max(team_a_win_prob, team_b_win_prob)}% win probability)")
                            
                            # Enhanced kill prediction display with team breakdown
                            st.write(f"💥 **Expected Total Kills:** {round(expected_kills, 1)}")
                            
                            # Add team-specific kill expectations
                            team_kills_col1, team_kills_col2 = st.columns(2)
                            with team_kills_col1:
                                a_kill_display = round(a_expected_kills, 1) if 'a_expected_kills' in locals() else "N/A"
                                st.write(f"**{team_a} Expected Kills:** {a_kill_display}")
                            with team_kills_col2:
                                b_kill_display = round(b_expected_kills, 1) if 'b_expected_kills' in locals() else "N/A"
                                st.write(f"**{team_b} Expected Kills:** {b_kill_display}")
                            
                            st.write(f"🔺 **Kill Spread:** {predicted_winner} wins by {kill_spread} kills.")
                            
                            # Function to describe pace
                            def get_pace_description(pace_value):
                                if pace_value > 1.3:
                                    return "Extremely Fast"
                                elif pace_value > 1.15:
                                    return "Very Fast"
                                elif pace_value > 1.05:
                                    return "Fast"
                                elif pace_value >= 0.95:
                                    return "Average"
                                elif pace_value >= 0.85:
                                    return "Slow"
                                elif pace_value >= 0.7:
                                    return "Very Slow"
                                else:
                                    return "Extremely Slow"
                            
                            # Calculate pace descriptions
                            team_a_pace_desc = get_pace_description(team_a_pace)
                            team_b_pace_desc = get_pace_description(team_b_pace)
                            match_pace_desc = get_pace_description(match_pace)
                            
                            # Display pace information
                            st.write(f"⚡ **Match Pace:** {match_pace_desc} ({round(match_pace, 2)}x average)")
                            
                            # Show individual team pace
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{team_a} Pace:** {team_a_pace_desc} ({round(team_a_pace, 2)}x)")
                                # Show KPM if available
                                if has_game_time_data and team_a_kpm > 0:
                                    st.write(f"**{team_a} KPM:** {round(team_a_kpm, 2)} kills/min")
                            with col2:
                                st.write(f"**{team_b} Pace:** {team_b_pace_desc} ({round(team_b_pace, 2)}x)")
                                # Show KPM if available
                                if has_game_time_data and team_b_kpm > 0:
                                    st.write(f"**{team_b} KPM:** {round(team_b_kpm, 2)} kills/min")
                            
                            # Calculate advanced match statistics
                            st.markdown("---")
                            st.write("### Advanced Match Statistics")
                            
                            # Calculate jungle control based on jungle matchup
                            jungle_advantage = "Even"
                            jungle_control = 50.0
                            for match in matchup_data:
                                if match["Role"] == "Jungle":
                                    if match["Advantage"] != "Even":
                                        jungle_advantage = match["Advantage"]
                                        # Calculate jungle control percentage
                                        if team_a in jungle_advantage:
                                            # Team A has jungle advantage
                                            score_diff = match["Score A"] - match["Score B"]
                                            jungle_control = 50 + min(20, score_diff / 20)
                                        else:
                                            # Team B has jungle advantage
                                            score_diff = match["Score B"] - match["Score A"]
                                            jungle_control = 50 - min(20, score_diff / 20)
                            
                            # Dragon control based on jungle control and bot lane advantage
                            bot_advantage = "Even"
                            support_advantage = "Even"
                            for match in matchup_data:
                                if match["Role"] == "Bot":
                                    bot_advantage = match["Advantage"]
                                elif match["Role"] == "Support":
                                    support_advantage = match["Advantage"]
                            
                            # Calculate dragon control
                            dragon_control = jungle_control
                            
                            # Adjust based on bot lane
                            if bot_advantage != "Even":
                                if team_a in bot_advantage and jungle_control >= 50:
                                    dragon_control += 5
                                elif team_a in bot_advantage and jungle_control < 50:
                                    dragon_control += 3
                                elif team_b in bot_advantage and jungle_control <= 50:
                                    dragon_control -= 5
                                elif team_b in bot_advantage and jungle_control > 50:
                                    dragon_control -= 3
                            
                            # Adjust based on support
                            if support_advantage != "Even":
                                if team_a in support_advantage and jungle_control >= 50:
                                    dragon_control += 3
                                elif team_a in support_advantage and jungle_control < 50:
                                    dragon_control += 2
                                elif team_b in support_advantage and jungle_control <= 50:
                                    dragon_control -= 3
                                elif team_b in support_advantage and jungle_control > 50:
                                    dragon_control -= 2
                            
                            # Normalize dragon control to 30-70% range
                            dragon_control = max(30, min(70, dragon_control))
                            
                            # Baron control (based on mid/top/jungle advantages)
                            baron_control = 50.0
                            for match in matchup_data:
                                if match["Role"] in ["Top", "Mid", "Jungle"]:
                                    if match["Advantage"] != "Even":
                                        role_weight = 1.0
                                        if match["Role"] == "Jungle":
                                            role_weight = 1.5
                                        
                                        if team_a in match["Advantage"]:
                                            diff = match["Score A"] - match["Score B"]
                                            baron_control += min(4, diff / 50) * role_weight
                                        else:
                                            diff = match["Score B"] - match["Score A"]
                                            baron_control -= min(4, diff / 50) * role_weight
                            
                            # Normalize baron control to 35-65% range
                            baron_control = max(35, min(65, baron_control))
                            
                            # First blood probability
                            first_blood_team = predicted_winner
                            first_blood_prob = 50 + (abs(team_a_win_prob - team_b_win_prob) / 4)
                            
                            # Adjust for jungle and mid early aggression
                            for match in matchup_data:
                                if match["Role"] in ["Jungle", "Mid"]:
                                    if match["Advantage"] != "Even":
                                        if team_a in match["Advantage"] and first_blood_team == team_a:
                                            first_blood_prob += 5
                                        elif team_b in match["Advantage"] and first_blood_team == team_b:
                                            first_blood_prob += 5
                            
                            # Cap first blood probability
                            first_blood_prob = min(75, first_blood_prob)
                            
                            # Display advanced stats
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"🐉 **Dragon Control:** {predicted_winner} ({round(max(dragon_control, 100-dragon_control), 1)}%)")
                                st.write(f"🧙 **Baron Control:** {predicted_winner} ({round(max(baron_control, 100-baron_control), 1)}%)")
                            with col2:
                                st.write(f"🩸 **First Blood:** {first_blood_team} ({round(first_blood_prob, 1)}%)")
                                st.write(f"⚔️ **Early Game Advantage:** {team_a if a_weighted_total > b_weighted_total else team_b}")
                            
                            # Game duration prediction using statistical approach
                            # Calculate expected game duration
                            if has_game_time_data and team_a_time > 0 and team_b_time > 0:
                                # If we have specific game time data for both teams, use their averages
                                # Teams with longer average games tend to play longer games against each other
                                base_duration = (team_a_time + team_b_time) / 2
                                
                                # Adjust for team kill rates (higher combined KPM = faster games)
                                if team_a_kpm > 0 and team_b_kpm > 0:
                                    combined_kpm = (team_a_kpm + team_b_kpm) / 2
                                    
                                    # Calculate average KPM across the league if we have that data
                                    if league_kpm_values and len(league_kpm_values) > 0:
                                        avg_kpm = sum(league_kpm_values) / len(league_kpm_values)
                                        # Higher than average KPM = shorter games
                                        kpm_factor = avg_kpm / combined_kpm if combined_kpm > 0 else 1.0
                                        # Apply a moderate adjustment based on KPM (max ~10% change)
                                        base_duration *= max(0.9, min(1.1, kpm_factor))
                            elif league_game_times and len(league_game_times) > 0:
                                # If we don't have team-specific times, use league average
                                avg_game_time = sum(league_game_times) / len(league_game_times)
                                base_duration = avg_game_time
                            else:
                                # Fallback to a reasonable average if no data
                                base_duration = 32.0 * 60  # League of Legends average game time in SECONDS
                            
                            # Competitiveness factor - closer games tend to be longer
                            # Use win probability to estimate competitiveness
                            win_prob_diff = abs(team_a_win_prob - team_b_win_prob) / 100
                            
                            # Statistically, very one-sided games tend to be 5-15% shorter
                            if win_prob_diff > 0.3:  # Significant skill gap
                                competitiveness_factor = 1.0 - (win_prob_diff - 0.3) * 0.3  # Max 9% reduction
                            else:  # Close matchup
                                competitiveness_factor = 1.0 + (0.3 - win_prob_diff) * 0.1  # Max 3% increase
                            
                            # Apply competitiveness factor
                            expected_duration = base_duration * competitiveness_factor
                            
                            # Add minimal natural variation (±1.5 minutes)
                            expected_duration += np.random.uniform(-60, 60)  # Convert to seconds (±1 minute)
                            
                            # Display with minutes and seconds
                            minutes = int(expected_duration // 60)
                            seconds = int(expected_duration % 60)
                            st.write(f"⏱️ **Expected Game Duration:** {minutes} minutes {seconds} seconds")

                            # Create a player comparison table
                            st.write("### Player Matchups by Role")
                            
                            # Get player rosters with improved matching
                            team_a_players = get_team_players(team_a, st.session_state.all_data)
                            team_b_players = get_team_players(team_b, st.session_state.all_data)
                            
                            # Create comparison table
                            matchup_data = []
                            
                            # Prepare for manual player selection fallback
                            if 'all_players_by_role' not in st.session_state:
                                st.session_state['all_players_by_role'] = {}
                            
                            all_players_by_role = {}
                            for role in roles:
                                # Check if we already have this data in session state
                                if role in st.session_state['all_players_by_role']:
                                    all_players_by_role[role] = st.session_state['all_players_by_role'][role]
                                    continue
                                
                                # Otherwise, load the data
                                all_players_by_role[role] = []
                                
                                # Show loading indicator
                                with st.spinner(f"Loading {role} player data (one-time operation)..."):
                                    # Collect all players for this role across leagues
                                    for league in leagues:
                                        try:
                                            player_data = get_player_data_with_fallback(league, role)
                                            if not isinstance(player_data, str) and not player_data.empty and "Player" in player_data.columns:
                                                for _, row in player_data.iterrows():
                                                    player_name = row["Player"]
                                                    player_score = row["Score"]
                                                    if pd.notna(player_name) and pd.notna(player_score):
                                                        all_players_by_role[role].append({
                                                            "name": player_name,
                                                            "score": player_score,
                                                            "league": league
                                                        })
                                        except Exception as e:
                                            print(f"Error loading {league} {role} player data: {str(e)}")
                                
                                # Cache the results for future use
                                st.session_state['all_players_by_role'][role] = all_players_by_role[role]
                            
                            # Count how many players we found automatically
                            auto_found_count_a = sum(1 for role in roles if team_a_players[role]["Player"] != "Unknown")
                            auto_found_count_b = sum(1 for role in roles if team_b_players[role]["Player"] != "Unknown")
                            
                            # Better player selection interface  
                            st.write("#### Player Matchups")
                            
                            # Show warning if we didn't find many players
                            if auto_found_count_a < 3 or auto_found_count_b < 3:
                                st.warning(f"Auto-identified {auto_found_count_a} players for {team_a} and {auto_found_count_b} players for {team_b}. Please select players using the dropdowns below.")
                            
                            # Add team league selectors to help narrow down player selection
                            st.write("##### Select Team Leagues")
                            col1, col2 = st.columns(2)
                            with col1:
                                team_a_league = st.selectbox(
                                    f"{team_a} League",
                                    options=["All Leagues"] + list(leagues.keys()),
                                    key=f"{team_a}_league"
                                )
                            with col2:
                                team_b_league = st.selectbox(
                                    f"{team_b} League",
                                    options=["All Leagues"] + list(leagues.keys()),
                                    key=f"{team_b}_league"
                                )
                            
                            # Create player selection for each role
                            for role in roles:
                                st.write(f"##### {role} Lane")
                                
                                # Create two columns for each team
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    # Filter players by league if selected
                                    filtered_players = all_players_by_role[role]
                                    if team_a_league != "All Leagues":
                                        filtered_players = [p for p in filtered_players if p["league"] == team_a_league]
                                        if not filtered_players:  # If no players in selected league, show all
                                            filtered_players = all_players_by_role[role]
                                    
                                    # Sort players by score for better dropdown experience
                                    sorted_players = sorted(filtered_players, key=lambda x: x["score"], reverse=True)
                                    player_names = [p["name"] for p in sorted_players]
                                    
                                    # Find index of auto-selected player if available
                                    default_idx = 0
                                    if team_a_players[role]["Player"] != "Unknown" and team_a_players[role]["Player"] in player_names:
                                        default_idx = player_names.index(team_a_players[role]["Player"])
                                    
                                    # Add team name to dropdown label for clarity
                                    selected_player_a = st.selectbox(
                                        f"{team_a} {role}", 
                                        options=player_names,
                                        index=default_idx,
                                        key=f"{team_a}_{role}"
                                    )
                                    
                                    # Update team_a_players with selected player
                                    selected_idx = player_names.index(selected_player_a)
                                    team_a_players[role]["Player"] = selected_player_a
                                    team_a_players[role]["Score"] = sorted_players[selected_idx]["score"]
                                    
                                    # Show selected player score
                                    st.write(f"Score: {round(team_a_players[role]['Score'], 1)}")
                                
                                with col_b:
                                    # Filter players by league if selected
                                    filtered_players = all_players_by_role[role]
                                    if team_b_league != "All Leagues":
                                        filtered_players = [p for p in filtered_players if p["league"] == team_b_league]
                                        if not filtered_players:  # If no players in selected league, show all
                                            filtered_players = all_players_by_role[role]
                                    
                                    # Sort players by score for better dropdown experience
                                    sorted_players = sorted(filtered_players, key=lambda x: x["score"], reverse=True)
                                    player_names = [p["name"] for p in sorted_players]
                                    
                                    # Find index of auto-selected player if available
                                    default_idx = 0
                                    if team_b_players[role]["Player"] != "Unknown" and team_b_players[role]["Player"] in player_names:
                                        default_idx = player_names.index(team_b_players[role]["Player"])
                                    
                                    # Add team name to dropdown label for clarity
                                    selected_player_b = st.selectbox(
                                        f"{team_b} {role}", 
                                        options=player_names,
                                        index=default_idx,
                                        key=f"{team_b}_{role}"
                                    )
                                    
                                    # Update team_b_players with selected player
                                    selected_idx = player_names.index(selected_player_b)
                                    team_b_players[role]["Player"] = selected_player_b
                                    team_b_players[role]["Score"] = sorted_players[selected_idx]["score"]
                                    
                                    # Show selected player score
                                    st.write(f"Score: {round(team_b_players[role]['Score'], 1)}")
                                
                                # Calculate score difference and determine advantage
                                a_player = team_a_players[role]["Player"]
                                a_score = team_a_players[role]["Score"]
                                b_player = team_b_players[role]["Player"]
                                b_score = team_b_players[role]["Score"]
                                
                                diff = a_score - b_score
                                if abs(diff) < 50:
                                    advantage = "Even"
                                else:
                                    advantage = f"{a_player if diff > 0 else b_player} (+{abs(round(diff, 1))})"
                                
                                # Add player comparison row
                                matchup_data.append({
                                    "Role": role,
                                    "Team A": a_player,
                                    "Score A": round(a_score, 1),
                                    "Team B": b_player,
                                    "Score B": round(b_score, 1),
                                    "Advantage": advantage
                                })
                            
                            # Display summary after all players are selected
                            st.write("### Matchup Summary")
                            
                            # Format player names for better clarity
                            formatted_matchup_data = []
                            for match in matchup_data:
                                formatted_match = match.copy()
                                formatted_match["Team A"] = format_player_name(match["Team A"], team_a)
                                formatted_match["Team B"] = format_player_name(match["Team B"], team_b)
                                
                                # Update advantage text with formatted names
                                if match["Advantage"] != "Even":
                                    if match["Team A"] in match["Advantage"]:
                                        # Use the formatted player name without team prefix
                                        formatted_match["Advantage"] = f"{formatted_match['Team A']} (+{abs(round(match['Score A'] - match['Score B'], 1))})"
                                    else:
                                        # Use the formatted player name without team prefix
                                        formatted_match["Advantage"] = f"{formatted_match['Team B']} (+{abs(round(match['Score B'] - match['Score A'], 1))})"
                                
                                formatted_matchup_data.append(formatted_match)
                            
                            # Convert to dataframe for better display
                            matchup_df = pd.DataFrame(formatted_matchup_data)
                            st.dataframe(matchup_df, use_container_width=True)
                            
                            # Show overall player score comparison
                            a_total = sum(team_a_players[role]["Score"] for role in roles)
                            b_total = sum(team_b_players[role]["Score"] for role in roles)
                            
                            # Define role weights (some roles have more impact on the game)
                            role_weights = {
                                "Top": 0.17,
                                "Jungle": 0.25,
                                "Mid": 0.23,
                                "Bot": 0.22,
                                "Support": 0.13
                            }
                            
                            # Calculate weighted player scores
                            a_weighted_total = sum(team_a_players[role]["Score"] * role_weights[role] for role in roles)
                            b_weighted_total = sum(team_b_players[role]["Score"] * role_weights[role] for role in roles)
                            
                            st.write(f"**Combined Player Scores:** {team_a}: {round(a_total, 1)} | {team_b}: {round(b_total, 1)}")
                            st.write(f"**Weighted Player Scores:** {team_a}: {round(a_weighted_total, 1)} | {team_b}: {round(b_weighted_total, 1)}")
                            
                            # Calculate individual role advantages based on final selections
                            a_advantages = sum(1 for match in matchup_data if match["Advantage"] != "Even" and match["Team A"] in match["Advantage"])
                            b_advantages = sum(1 for match in matchup_data if match["Advantage"] != "Even" and match["Team B"] in match["Advantage"])
                            st.write(f"**Role Advantages:** {team_a}: {a_advantages} | {team_b}: {b_advantages} | Even: {5-a_advantages-b_advantages}")

                        except Exception as e:
                            st.error(f"⚠️ Error calculating match details: {e}")
                            st.write("Team statistics may be missing or in an unexpected format.")

                except Exception as e:
                    st.error(f"⚠️ Error processing team data: {e}")
                    st.write("Debug info:")
                    st.write(f"Team A columns: {team_a_data.columns.tolist() if 'team_a_data' in locals() else 'Not available'}")
                    st.write(f"Team B columns: {team_b_data.columns.tolist() if 'team_b_data' in locals() else 'Not available'}")
            else:
                st.warning("Please select both teams to generate a match prediction.")
        else:
            st.error("Team data is missing the 'Score' column. Available columns: " + ", ".join(global_team_df.columns))

# Add a refresh button at the bottom of the sidebar to manually refresh the data
st.sidebar.markdown("---")
if st.sidebar.button("Refresh All Data"):
    # Clear session state variables
    st.session_state.pop('data_loaded', None)
    st.session_state.pop('all_data', None)
    st.session_state.pop('global_team_df', None)
    st.session_state.pop('loaded_leagues', None)
    st.session_state.pop('failed_leagues', None)
    st.session_state.pop('all_players_by_role', None)
    
    # Clear all cache data functions explicitly
    fetch_raw_sheet_data.clear()
    fetch_all_sheet_tabs.clear()
    get_player_data.clear()
    get_team_data.clear()
    load_all_data.clear()
    load_all_data_with_workers.clear()
    get_team_players.clear()
    
    # Also clear any disk cache
    try:
        cache_dir = ".sheet_cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        st.warning(f"Could not clear disk cache: {e}")
    
    # Force a complete rerun
    st.rerun()
