import re
from io import StringIO

import pandas as pd
import requests


def query_crts_cone(ra, dec, radius_arcmin=0.16, output_format="csv", detail="short"):
    """
    Query CRTS database using POST request.

    Parameters
    ----------
    ra : float
        Right Ascension in decimal degrees (J2000)
    dec : float
        Declination in decimal degrees (J2000)
    radius_arcmin : float
        Search radius in arcminutes (max ~3.0)
    output_format : str
        'csv', 'votable', or 'html'
    detail : str
        'short' or 'long' (amount of data returned)

    Returns
    -------
    response : requests.Response object
    """
    url = "http://nunuku.caltech.edu/cgi-bin/getcssconedb_release_img.cgi"

    # Format coordinates as string with whitespace (tab or space)
    radec_str = f"{ra}  {dec}"

    data = {
        "RADec": radec_str,  # Key parameter: combined RA Dec
        "Rad": str(radius_arcmin),
        "IMG": "nun",  # 'nun' = None, 'dss' = DSS, 'sds' = SDSS
        "DB": "photcat",  # or 'orphancat'
        ".submit": "Submit",
        "OUT": output_format,  # 'csv', 'votable', or 'html'
        "SHORT": detail,  # 'short' or 'long'
        "PLOT": "plot",  # 'plot' or 'hide'
    }

    print(f"Querying CRTS: RA={ra}, Dec={dec}, radius={radius_arcmin} arcmin")

    try:
        # POST request (not GET!)
        response = requests.post(url, data=data, timeout=60)

        if response.status_code != 200:
            print(f"HTTP Error {response.status_code}")
            return None

        # Check for errors in response
        if "You must input coordinates!" in response.text:
            print("Error: Coordinate format not recognized")
            return None
        if "There were 0 lines" in response.text:
            print("No sources found in search region")
            return response
        if "var dataSet" in response.text or output_format == "csv":
            print("Query successful!")
            return response
        print("Query completed (check response)")
        return response

    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_crts_html(html_content):
    """
    Parse HTML response to extract lightcurve data from JavaScript.

    Returns
    -------
    dict : {object_name: DataFrame} with MJD, Magnitude, Error columns
    """
    # Find all JavaScript dataSet variables
    pattern = r'var dataSet(\d+)\s*=\s*\{\s*label:\s*"([^"]+)"[^{]*?data:\s*(\[\[.*?\]\])'
    matches = re.findall(pattern, html_content, re.DOTALL)

    if not matches:
        print("No JavaScript lightcurve data found")
        print("Trying to parse HTML table instead...")
        return parse_crts_html_table(html_content)

    lightcurves = {}

    for match in matches:
        dataset_num, object_name, data_str = match

        # Extract individual observations [MJD, mag, error]
        # Pattern matches: [number, number, number]
        obs_pattern = r"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]"
        observations = re.findall(obs_pattern, data_str)

        if not observations:
            print(f"Warning: No observations parsed for {object_name}")
            print(f"Data string length: {len(data_str)}")
            print(f"First 200 chars: {data_str[:200]}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(observations, columns=["MJD", "Magnitude", "Error"])
        df = df.astype(float)

        lightcurves[object_name] = df

        print(f"{object_name}:")
        print(f"  Observations: {len(df)}")
        print(f"  MJD range: {df['MJD'].min():.1f} - {df['MJD'].max():.1f}")
        print(f"  Mag range: {df['Magnitude'].min():.2f} - {df['Magnitude'].max():.2f}")
        print(f"  Mean mag: {df['Magnitude'].mean():.2f} ± {df['Magnitude'].std():.2f}")

    return lightcurves


def parse_crts_html_table(html_content):
    """
    Alternative parser: Extract lightcurve data from the HTML table.
    Falls back to this if JavaScript parsing fails.

    Note: CRTS HTML tables are malformed (missing </td> tags),
    so we use regex instead of BeautifulSoup.
    """
    # The HTML table has rows like:
    # <tr><td>OBJ_ID<td>Mag<td>Magerr<td>RA<td>Dec<td>MJD</tr>

    # Extract all table rows with photometry data
    pattern = r"<tr><td>(\d+)<td>([\d.]+)<td>([\d.]+)<td>[\d.]+<td>[\d.]+<td>([\d.]+)</tr>"
    matches = re.findall(pattern, html_content)

    if not matches:
        print("Could not parse HTML table - no photometry data found")
        return {}

    # Convert to DataFrame
    data = [[obj_id, float(mjd), float(mag), float(magerr)] for obj_id, mag, magerr, mjd in matches]

    df_all = pd.DataFrame(data, columns=["ObjID", "MJD", "Magnitude", "Error"])

    # Group by object ID
    lightcurves = {}
    for obj_id in df_all["ObjID"].unique():
        df_obj = df_all[df_all["ObjID"] == obj_id][["MJD", "Magnitude", "Error"]].copy()
        df_obj = df_obj.sort_values("MJD").reset_index(drop=True)
        lightcurves[obj_id] = df_obj

        print(f"{obj_id}:")
        print(f"  Observations: {len(df_obj)}")
        print(f"  MJD range: {df_obj['MJD'].min():.1f} - {df_obj['MJD'].max():.1f}")
        print(f"  Mag range: {df_obj['Magnitude'].min():.2f} - {df_obj['Magnitude'].max():.2f}")

    return lightcurves


def parse_crts_csv(csv_text):
    """
    Parse CSV response from CRTS.
    """
    # Check if we got HTML instead of CSV
    if csv_text.strip().startswith("<!DOCTYPE") or csv_text.strip().startswith("<html"):
        print("Response is HTML, not CSV. Extracting CSV download link...")
        csv_link = extract_csv_download_link(csv_text)
        if csv_link:
            print(f"CSV download link found: {csv_link}")
            print("Downloading CSV...")
            try:
                response = requests.get(csv_link, timeout=30)
                if response.ok:
                    csv_text = response.text
                    print("CSV downloaded successfully")
                else:
                    print(f"Failed to download CSV: HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading CSV: {e}")
                return None
        else:
            print("No CSV download link found")
            return None

    try:
        df = pd.read_csv(StringIO(csv_text), comment="#")
        print(f"Parsed CSV with {len(df)} rows and {len(df.columns)} columns")
        if len(df) > 0:
            print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        print("First 1000 chars of CSV:")
        print(csv_text[:1000])
        return None


def extract_csv_download_link(html_content):
    """Extract temporary CSV download link from HTML response."""
    pattern = r"(http://nunuku\.caltech\.edu/DataRelease/upload/result_web_file\w+\.csv)"
    match = re.search(pattern, html_content)
    return match.group(1) if match else None


def get_crts_csv_data(ra, dec, radius_arcmin=0.16, detail="long"):
    """
    Convenience function to get CRTS data directly as a DataFrame.
    This queries CRTS and extracts the CSV download link.

    Parameters
    ----------
    ra : float
        Right Ascension in decimal degrees
    dec : float
        Declination in decimal degrees
    radius_arcmin : float
        Search radius in arcminutes
    detail : str
        'short' or 'long'

    Returns
    -------
    DataFrame : Combined lightcurve data for all objects in the search region
    """
    response = query_crts_cone(ra, dec, radius_arcmin, output_format="csv", detail=detail)

    if not response:
        return None

    csv_link = extract_csv_download_link(response.text)

    if not csv_link:
        print("No CSV download link found. Trying to parse HTML for lightcurves...")
        # Fall back to parsing HTML
        lcs = parse_crts_html(response.text)
        if lcs:
            # Combine all lightcurves into one DataFrame
            all_data = []
            for obj_name, df in lcs.items():
                df["Object"] = obj_name
                all_data.append(df)
            if all_data:
                return pd.concat(all_data, ignore_index=True)
        return None

    print(f"Downloading CSV from: {csv_link}")
    try:
        csv_response = requests.get(csv_link, timeout=30)
        if csv_response.ok:
            df = pd.read_csv(StringIO(csv_response.text), comment="#")
            print(f"Got {len(df)} rows from CSV")
            return df
        print(f"Failed to download CSV: HTTP {csv_response.status_code}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def query_crts_batch(positions, radius_arcmin=0.16, output_format="csv"):
    """
    Query multiple positions.

    Parameters
    ----------
    positions : list of tuples
        [(ra1, dec1), (ra2, dec2), ...]
    radius_arcmin : float
        Search radius in arcminutes
    output_format : str
        Output format

    Returns
    -------
    list : List of response objects
    """
    results = []

    for i, (ra, dec) in enumerate(positions, 1):
        print(f"Query {i}/{len(positions)}")
        response = query_crts_cone(ra, dec, radius_arcmin, output_format)
        if response:
            results.append({"position": (ra, dec), "response": response})

    return results


# Example usage
if __name__ == "__main__":
    ra = 160.7980
    dec = 11.3972
    radius = 0.16  # arcminutes

    df = get_crts_csv_data(ra, dec, radius_arcmin=radius, detail="short")

    if df is not None and len(df) > 0:
        print(f"Got {len(df)} observations")
        print(f"Columns: {list(df.columns)}")
        print("First 10 rows:")
        print(df.head(10))

        # Save to file
        df.to_csv("crts_lightcurve_data.csv", index=False)
        print("Saved all data to: crts_lightcurve_data.csv")
