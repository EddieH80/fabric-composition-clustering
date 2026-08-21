import re
import json
import requests
import csv
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Constants
BRANDS = {
    "HANNOVAE": "https://www.hannovae.com/products.json",
    "PERSONSOUL": "https://www.personsoul.com/products.json",
    "PETERDO": "https://www.peterdo.net/products.json",
    "SUNDAE SCHOOL": "https://sundae.school/products.json",
    "NUWA HANFU": "https://nuwahanfu.com/products.json",
    "PH5": "https://ph5.com/products.json",
    "PETITE STUDIO": "https://www.petitestudionyc.com/products.json",
    "AKASHI KAMA": "https://www.akashi-kama.com/products.json",
    "KIRIKO MADE": "https://kirikomade.com/products.json",
    "NI DE MAMA": "https://nidemama.com/products.json",
    "ANNA SUI": "https://annasui.com/products.json",
    "SANDY LIANG": "https://www.sandyliang.info/products.json",
    "ANTA": "https://anta.com/products.json",
    "CHINASQUAD": "https://chinasquad.com/products.json",
    "ADEAM": "https://www.adeam.com/products.json",
    "DAWANG": "https://www.dawangnewyork.com/products.json",
    "YANYAN KNITS": "https://yanyanknits.com/products.json",
    "TAE PARK": "https://www.tae-park.com/products.json",
    "HYEIN SEO": "https://hyeinseo.com/products.json",
    "BAD BINCH TONGTONG": "https://badbinch.com/products.json",
    "LUU DAN": "https://luu-dan.com/products.json",
    "KARTIK RESEARCH": "https://www.kartikresearch.com/products.json",
    "TAIGA TAKAHASHI": "https://taigatakahashi.com/en/products.json",
    "DOUBLET": "https://shop.doublet-jp.com/en/products.json",
    "PENG TAI": "https://uj-ng.com/collections/women-peng-tai/products.json",
    "TIRADOS": "https://www.tirados.co/products.json",
    "FRIZMWORKS": "https://frizmworks.eu/products.json"
}

RELEVANT_DATA = ["BRAND", "PRODUCT NAME", "PRODUCT TYPE", "ID", "PRICE", "FABRIC COMPOSITION"]
WORKING_DIRECTORY = "C:\\Users\\eddie\\Desktop\\Coding\\Python\\fabric-composition-clustering"
OUTPUT_CSV = "clothing_brand_products.csv"
CLUSTERED_CSV = "clothing_brand_products_clustered.csv"
SCRIPT_NAME = "scrape_and_cluster"
N_CLUSTERS = 4
MAX_K = 11

# Extract fabric composition information (assumed to be in "body_html" field) using regex
def extract_fabric_composition(body_html: str) -> str:
    if not body_html:
        return "N/A"
    html_parser = BeautifulSoup(body_html, "html.parser")
    extracted_html = html_parser.get_text(separator=" ")
    # TODO: Patternmatching is not super robust (matches ), could use more sophisticated method like LLM's to handle this
    pattern = r'\b(\d+(\.\d+)?%\s*\w+(?:\s+\w+){0,1}+(?:[\s,/]+\d+%\s*\w+)*)'
    match = re.search(pattern, extracted_html, re.IGNORECASE)
    if match:
        return match.group().strip()
    return "N/A"

# Scrape relevant product data from each brand's public "/products.json" endpoint and save it in a CSV file
def scrape_brand_data(brands: dict, output_csv_path: str) -> None:
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RELEVANT_DATA)

        for brand, url in brands.items():
            response = requests.get(url)
            if response.status_code == 200:
                try:
                    response.json()  # Check if the response is valid JSON
                except json.JSONDecodeError:
                    print(f"Invalid JSON response for brand: {brand}")
                    continue
                for product in response.json()["products"]:
                    try:
                        # If RELEVANT_DATA fields are changed, this must be changed too
                        writer.writerow([brand, product["title"], product["product_type"], product["id"], product["variants"]
                                     [0]["price"], extract_fabric_composition(product["body_html"])])
                    except KeyError as e:
                        print("Missing field expected in product data:", e)          
            else:
                print(f"Failed to fetch data for {brand}, got status code: {response.status_code})")

# Recognized textile fiber vocabulary: canonical name -> regex patterns that identify it.
# This works as an ALLOWLIST rather than a blocklist. Real-world textile fibers are a
# small, well-documented, essentially closed set, so checking extracted text against
# this vocabulary scales far better than trying to blacklist every possible junk phrase
# a page's body_html might contain (e.g. "Designed in Italy" getting mis-captured as a
# material). Anything that matches no known fiber is treated as noise and dropped - no
# manual junk list to maintain.
#
# More specific/premium variants are listed BEFORE their generic parent so they're
# matched first and kept as distinct categories (e.g. "pima cotton" stays separate
# from plain "cotton"; "merino wool" stays separate from plain "wool").
FIBER_VOCABULARY = [
    ("pima cotton",        [r"\bpima\b"]),
    ("egyptian cotton",    [r"\begyptian\b"]),
    ("organic cotton",     [r"\borganic\b.*\bcotton\b", r"\bcotton\b.*\borganic\b"]),
    ("combed cotton",      [r"\bcombed\b", r"\bairlume\b"]),
    ("cotton",             [r"\bcotton\b"]),

    ("merino wool",        [r"\bmerino\b"]),
    ("cashmere",           [r"\bcashmere\b"]),
    ("mohair",             [r"\bmohair\b"]),
    ("lambswool",          [r"\blambswool\b", r"lamb.?s\s*wool"]),
    ("alpaca",             [r"\balpaca\b"]),
    ("camel hair",         [r"\bcamel\b"]),
    ("yak wool",           [r"\byak\b"]),
    ("wool",               [r"\bwool\b"]),

    ("silk",               [r"\bsilk\b", r"\bmulberry\b"]),
    ("linen",              [r"\blinen\b"]),
    ("ramie",              [r"\bramie\b"]),
    ("hemp",               [r"\bhemp\b"]),

    ("recycled polyester", [r"recycled.*polyester"]),
    ("polyester",          [r"\bpolyester\b", r"\bpoly\b"]),
    ("recycled nylon",     [r"recycled.*nylon"]),
    ("nylon",              [r"\bnylon\b"]),
    ("spandex",            [r"\bspandex\b", r"\belastane\b"]),
    ("polyamide",          [r"\bpolyamide\b"]),
    ("polyurethane",       [r"\bpolyurethane\b"]),
    ("polyethylene",       [r"\bpolyethylene\b"]),
    ("acrylic",            [r"\bacryl"]),
    ("acetate",            [r"\bacetate\b"]),

    ("tencel/lyocell",     [r"\btencel\b", r"\blyocell\b"]),
    ("modal",              [r"\bmodal\b"]),
    ("viscose/rayon",      [r"\bviscose\b", r"\brayon\b", r"\becovero\b", r"\bcupro\b"]),

    ("leather",            [r"\bleather\b", r"\bnappa\b"]),
]


def normalize_material_name(raw_name: str):
    """Maps a messy, regex-extracted material name to a canonical fiber name by
    checking it against a known textile-fiber vocabulary. Returns None if no known
    fiber is found in the text - meaning it's treated as noise (e.g. text captured
    from unrelated body copy like "Designed in Italy") rather than an actual material.
    No manual junk list required: anything outside the recognized vocabulary is
    automatically dropped."""
    name = re.sub(r"\s+", " ", raw_name.lower().strip())

    for canonical, patterns in FIBER_VOCABULARY:
        if any(re.search(p, name) for p in patterns):
            return canonical

    return None  # no recognized fiber found in this text -> treat as noise


def reformat_fabric_composition(fabric_str: str) -> dict:
    # Converts format from '69% Cotton, 25% Polyamid, 6% Elastane' → {'cotton': 69, …}
    # Validates each extracted fragment against a known fiber vocabulary, merging
    # true synonyms (e.g. "elastane" -> "spandex") while keeping premium/distinct
    # subtypes separate (e.g. "pima cotton" != "cotton") and silently dropping
    # fragments that aren't recognizable fiber names at all.
    if not fabric_str or fabric_str == "N/A" or not isinstance(fabric_str, str):
        print("fabric_str is not in the expected format: ", fabric_str)
        return {}
    pattern = r'(\d+)%\s*([\w\s]+?)(?=,|$)'
    matches = re.findall(pattern, fabric_str, re.IGNORECASE)

    result = {}
    for percent, raw_material in matches:
        canonical = normalize_material_name(raw_material)
        if canonical is None:
            continue
        result[canonical] = result.get(canonical, 0) + int(percent)
    return result

# Returns (X, material_names) where X is the matrix we will use for the clustering
def build_material_matrix(df: pd.DataFrame) -> tuple:
    df["fabric_parsed"] = df["FABRIC COMPOSITION"].apply(reformat_fabric_composition)
    # Create set of all materials used in products for all brands
    all_materials = set()
    df["fabric_parsed"].apply(lambda x: all_materials.update(x.keys()))

    for material in all_materials:
        df[material] = df["fabric_parsed"].apply(lambda x: x.get(material, 0))

    material_names = list(all_materials)
    X = df[material_names].fillna(0)
    return X, material_names

# Plots the elbow graph to determine the optimal number of clusters based on the WCSS (within-cluster sum of squares)
def plot_elbow(X: pd.DataFrame, max_k: int = MAX_K) -> None:
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++",random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("The Elbow Method (determining optimal k)")
    plt.savefig(WORKING_DIRECTORY + "\\elbow_plot.png")
    plt.show()
    print("Plot saved as elbow_plot.png\n")

# Runs KMeans clustering on material composition data and prints the breakdown of products in each cluster
def run_kmeans_clustering(df: pd.DataFrame, X, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    print(f"Cluster breakdown ({n_clusters} clusters):")
    for cluster_id in range(n_clusters):
        products= df[df["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(products)} products):")
        print(products[["BRAND", "PRODUCT NAME", "FABRIC COMPOSITION"]].to_string(index=False))

    return df

# Plots the clusters in 2D by using PCA to reduce the dimensionality of the material composition data
def visualize_clusters(df: pd.DataFrame, X) -> None:
    pca = PCA(n_components=2)
    cluster_centers = pca.fit_transform(X)

    # Label each axis with its top-loading material and the variance it explains
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"])
    pc1_label = loadings["PC1"].abs().idxmax()
    pc2_label = loadings["PC2"].abs().idxmax()
    var = pca.explained_variance_ratio_

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=df["cluster"])
    plt.title("Product Clusters by Fabric Composition (PCA)")
    plt.xlabel(f"PCA Component 1 (~{pc1_label}, {var[0]*100:.0f}% var)")
    plt.ylabel(f"PCA Component 2 (~{pc2_label}, {var[1]*100:.0f}% var)")
    plt.savefig(WORKING_DIRECTORY + "\\cluster_plot.png")
    plt.show()
    print(f"Explained variance: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%, total={var.sum()*100:.1f}%")
    print("Plot saved as cluster_plot.png\n")

# Creates a PDF file containig all code from this file
def create_pdf() -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", fname=WORKING_DIRECTORY + "\\DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=10)

    with open(WORKING_DIRECTORY + "\\" + SCRIPT_NAME + ".py", encoding="utf-8", mode="r") as file:
        for line in file:
            pdf.multi_cell(0, 2, txt=line, align="L")

    pdf.output(WORKING_DIRECTORY + "\\" + SCRIPT_NAME + ".pdf")
    print("Code saved as PDF file in '" + SCRIPT_NAME + ".pdf'")

def main():
    # Data scraping and wrangling
    scrape_brand_data(BRANDS, WORKING_DIRECTORY + '\\' + OUTPUT_CSV)
    df = pd.read_csv(WORKING_DIRECTORY + '\\' + OUTPUT_CSV, encoding="utf-8-sig")
    df["FABRIC COMPOSITION"] = df["FABRIC COMPOSITION"].fillna("N/A").astype(str)
    df = df[df["FABRIC COMPOSITION"] != "N/A"].reset_index(drop=True)
    print(f"{len(df)} products with fabric data available for clustering.\n")
    if len(df) < 2:
        print("Not enough fabric data to cluster. Exiting...")
        return

    # Clustering
    X, material_names = build_material_matrix(df)
    print(f"List of materials used: {', '.join(sorted(material_names))}\n")
    plot_elbow(X)
    df = run_kmeans_clustering(df, X, n_clusters=N_CLUSTERS)
    visualize_clusters(df, X)
    df.drop(columns=["fabric_parsed"], errors="ignore").to_csv(WORKING_DIRECTORY + '\\' + CLUSTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"Clustered data saved to {WORKING_DIRECTORY + '\\' + CLUSTERED_CSV}")

    # # Save code as PDF
    # create_pdf()

if __name__ == "__main__":
    main()
