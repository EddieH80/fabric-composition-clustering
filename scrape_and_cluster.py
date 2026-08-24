import re
import json
import requests
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.ticker as mticker

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

    df["premium_fiber_pct"] = df["fabric_parsed"].apply(compute_premium_fiber_pct)

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

# Fibers considered "premium" for the purposes of a derived quality signal - long-staple
# cottons, fine animal fibers, and other fibers generally priced/marketed at a premium
# over their generic counterparts (plain cotton, polyester, nylon, etc.)
PREMIUM_FIBERS = {
    "pima cotton", "egyptian cotton", "combed cotton", "organic cotton",
    "merino wool", "cashmere", "mohair", "alpaca", "camel hair", "yak wool",
    "silk", "leather", "tencel/lyocell",
}

def compute_premium_fiber_pct(fabric_parsed: dict) -> float:
    """Sums the % composition of any fiber in PREMIUM_FIBERS for a single product."""
    return sum(pct for fiber, pct in fabric_parsed.items() if fiber in PREMIUM_FIBERS)

# --- Product type normalization ---------------------------------------------

# Same allowlist approach as FIBER_VOCABULARY: raw PRODUCT TYPE values are wildly
# inconsistent (casing, plurals, brand-specific taxonomies, delivery-season labels
# like "Fall 2026- Delivery 2"). Mapping to a known, bounded set of apparel
# categories scales better than trying to normalize casing/plurals/synonyms by hand.
PRODUCT_TYPE_VOCABULARY = [
    ("Dresses",           [r"\bdress"]),
    ("Outerwear",         [r"\bjacket", r"\bblazer\b", r"\bvest\b", r"\bouterwear\b",
                            r"\bcoat", r"\bnoragi\b", r"\bhappi\b", r"\bdougi\b"]),
    ("Knitwear & Sweats",  [r"\bknitwear\b", r"\bsweater\b", r"\bhoodie\b",
                             r"\bsweatshirt\b", r"\bfleece\b"]),
    ("Bottoms",            [r"\bpants?\b", r"\bjeans?\b", r"\bshorts?\b", r"\bskirt",
                             r"\bsweatpants?\b", r"\bbottoms?\b"]),
    ("Tops",               [r"\bshirt", r"\bblouse", r"\btee\b", r"\bt-?shirt",
                             r"\btank\b", r"\bpolo\b", r"\btops?\b"]),
    ("Footwear",           [r"\bshoes?\b", r"\bslides?\b", r"\bsneaker",
                             r"\bbasketball shoes\b", r"\brunning shoes\b"]),
    ("Bags",               [r"\bbag\b", r"\btote\b", r"\bduffle\b", r"\bclutch\b",
                             r"\bhandbag\b", r"\bwallet\b"]),
    ("Jewelry",            [r"\bjewelry\b", r"\bnecklace\b", r"\bearring",
                             r"\brings?\b", r"\bbracelet\b", r"\bpendant\b"]),
    ("Swim & Intimates",   [r"\bswimwear\b", r"\bbra\b", r"\bbriefs?\b"]),
    ("Accessories",        [r"\baccessor", r"\bbelt\b", r"\bgloves?\b", r"\bcap\b",
                             r"\bhats?\b", r"\bbeanie\b", r"\bbucket hat\b",
                             r"\bhair accessories\b", r"\bscarf\b", r"\bheadband\b"]),
    ("Non-Apparel",        [r"\bkitchen\b", r"\bcosmetics\b", r"\bsticker\b",
                             r"\bcup\b", r"\bfuroshiki\b", r"\bbeauty\b"]),
]


def normalize_product_type(raw_type: str):
    """Maps a raw PRODUCT TYPE value to a canonical apparel category. Returns None
    for values that don't match any known category (season/delivery labels, stray
    junk, blank values) so they're excluded from category-based analysis rather
    than treated as their own noisy one-off category."""
    if not isinstance(raw_type, str) or not raw_type.strip():
        return None
    name = raw_type.lower().strip()
    for canonical, patterns in PRODUCT_TYPE_VOCABULARY:
        if any(re.search(p, name) for p in patterns):
            return canonical
    return None


# --- Price cleanup ------------------------------------------------------------

# Some brands price in a currency other than USD (their storefront's base
# currency), which corrupts any cross-brand price comparison if left as-is.
# Flag any brand whose median price is drastically higher than the global median
# rather than hardcoding brand names, so this generalizes if you add brands later.
CURRENCY_OUTLIER_THRESHOLD = 15
ASSUMED_FX_RATE = 1300  # approx KRW->USD; verify actual rate before trusting this

# Some brands list a placeholder price for "not for sale" items (e.g. samples,
# giveaways) rather than omitting a price entirely. Filter these out.
PLACEHOLDER_PRICE_SENTINELS = {9999, 99999, 0}


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Flags and roughly converts currency-outlier brands, and drops placeholder
    'not for sale' prices, into a new PRICE_USD column."""
    df = df.copy()
    df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")
    df = df[~df["PRICE"].isin(PLACEHOLDER_PRICE_SENTINELS)]
    df = df.dropna(subset=["PRICE"]).reset_index(drop=True)

    global_median = df["PRICE"].median()
    brand_medians = df.groupby("BRAND")["PRICE"].median()
    outlier_brands = brand_medians[brand_medians > global_median * CURRENCY_OUTLIER_THRESHOLD].index.tolist()

    df["PRICE_USD"] = df["PRICE"]
    if outlier_brands:
        print(f"Currency scale outliers detected (likely non-USD): {outlier_brands}")
        print(f"Applying assumed conversion of /{ASSUMED_FX_RATE} - verify the real rate before trusting this.\n")
        mask = df["BRAND"].isin(outlier_brands)
        df.loc[mask, "PRICE_USD"] = df.loc[mask, "PRICE"] / ASSUMED_FX_RATE

    return df


# --- Price tier clustering -----------------------------------------------------

MIN_PRODUCTS_FOR_PRICE_CLUSTERING = 8
PRICE_TIERS = 3  # Budget / Mid-range / Premium


def cluster_prices_by_type(df: pd.DataFrame, n_tiers: int = PRICE_TIERS,
                            min_products: int = MIN_PRODUCTS_FOR_PRICE_CLUSTERING) -> pd.DataFrame:
    """For each normalized product type with enough data, runs K-Means on
    log-transformed price (log handles the right-skew typical of pricing data) to
    split products into price tiers, relabeled in ascending price order so labels
    are meaningful (Budget/Mid-range/Premium) instead of arbitrary cluster IDs."""
    df = df.copy()
    df["price_tier"] = None

    for ptype, group in df.groupby("PRODUCT_TYPE_NORMALIZED"):
        if ptype is None or len(group) < min_products:
            continue
        prices = group["PRICE_USD"].dropna()
        if len(prices) < min_products:
            continue

        log_prices = np.log1p(prices.values).reshape(-1, 1)
        k = min(n_tiers, prices.nunique())
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(log_prices)

        cluster_means = pd.Series(kmeans.cluster_centers_.flatten(), index=range(k))
        rank_order = cluster_means.sort_values().index.tolist()
        tier_names = ["Budget", "Mid-range", "Premium", "Luxury"][:k]
        label_map = {old_label: tier_names[rank] for rank, old_label in enumerate(rank_order)}

        df.loc[prices.index, "price_tier"] = [label_map[l] for l in raw_labels]

    return df


def visualize_price_tiers(df: pd.DataFrame) -> None:
    """Boxplot of USD price by normalized product type, sorted by median price,
    with individual products scattered on top (colored by K-Means tier), the
    median labeled directly on each box, and sample size shown per category."""
    plotted = df.dropna(subset=["PRODUCT_TYPE_NORMALIZED", "price_tier"])
    if plotted.empty:
        print("No product types had enough data to cluster into price tiers.")
        return

    # sort categories cheapest -> priciest so the plot reads as a progression
    categories = sorted(
        plotted["PRODUCT_TYPE_NORMALIZED"].unique(),
        key=lambda c: plotted[plotted["PRODUCT_TYPE_NORMALIZED"] == c]["PRICE_USD"].median()
    )
    tier_colors = {"Budget": "tab:blue", "Mid-range": "tab:orange",
                    "Premium": "tab:green", "Luxury": "tab:red"}

    fig, ax = plt.subplots(figsize=(13, 7))
    box_data = [plotted[plotted["PRODUCT_TYPE_NORMALIZED"] == cat]["PRICE_USD"].values
                for cat in categories]
    bp = ax.boxplot(box_data, tick_labels=categories, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("whitesmoke")
        patch.set_edgecolor("gray")

    for i, cat in enumerate(categories, start=1):
        sub = plotted[plotted["PRODUCT_TYPE_NORMALIZED"] == cat]
        jitter = np.random.normal(loc=i, scale=0.06, size=len(sub))
        colors = sub["price_tier"].map(tier_colors)
        ax.scatter(jitter, sub["PRICE_USD"], c=colors, alpha=0.65, s=22,
                   zorder=3, edgecolors="none")

        # label the median directly next to its box
        median = sub["PRICE_USD"].median()
        ax.annotate(f"${median:,.0f}", xy=(i, median), xytext=(i + 0.32, median),
                    fontsize=9, fontweight="bold", va="center", color="black")

        # sample size under each category's tick label
        ax.text(i, -0.06, f"n={len(sub)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="dimgray")

    ax.set_yscale("log")
    # Plain base-10 numbers (100, 1,000) instead of scientific "10^2" notation
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())  # hides cluttered minor-tick labels
    ax.set_ylabel("Price (USD, log scale)")
    ax.set_title("Price Distribution & Tiers by Product Type\n"
                 "(sorted by median price; dot color = K-Means tier)")
    ax.set_xticklabels(categories, rotation=35, ha="right")

    legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                                  markersize=8, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=legend_handles, title="Price tier", loc="upper left")

    plt.tight_layout()
    plt.savefig(WORKING_DIRECTORY + "\\price_tiers_by_type.png", dpi=130)
    plt.show()
    print("Plot saved as price_tiers_by_type.png\n")

    print("Price tier summary by product type:")
    summary = plotted.groupby(["PRODUCT_TYPE_NORMALIZED", "price_tier"])["PRICE_USD"] \
                      .agg(["count", "mean", "min", "max"])
    print(summary)

# --- Supervised price prediction ------------------------------------------

MIN_PRODUCTS_PER_TYPE_FOR_MODEL = 10   # drop categories too small to model reliably
MIN_MATERIAL_PRESENCE = 0.05            # keep a material only if it appears in >=5% of products


def build_price_model_dataset(df: pd.DataFrame, material_names: list) -> tuple:
    """Assembles the feature matrix (X) and target (y, raw USD price) for price
    prediction: normalized fiber percentages + premium_fiber_pct + one-hot encoded
    product type. Takes the material column names explicitly (from
    build_material_matrix) rather than inferring them by excluding known
    non-material columns, since df carries many other columns (PRODUCT NAME, ID,
    FABRIC COMPOSITION, cluster, price_tier, etc.) that a fragile exclusion list
    would otherwise sweep in. Drops product types with too few products to model
    reliably and drops materials too rare to provide reliable signal."""
    counts = df["PRODUCT_TYPE_NORMALIZED"].value_counts()
    keep_types = counts[counts >= MIN_PRODUCTS_PER_TYPE_FOR_MODEL].index
    df = df[df["PRODUCT_TYPE_NORMALIZED"].isin(keep_types)].reset_index(drop=True)

    type_dummies = pd.get_dummies(df["PRODUCT_TYPE_NORMALIZED"], prefix="type")
    presence = (df[material_names] > 0).mean()
    material_cols = presence[presence >= MIN_MATERIAL_PRESENCE].index.tolist()

    X = pd.concat([df[material_cols], df[["premium_fiber_pct"]], type_dummies], axis=1)
    y = df["PRICE_USD"].values
    return X, y


def evaluate_price_model(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> None:
    """5-fold cross-validated comparison of a Random Forest against a fair
    per-category-median baseline (baseline is computed from the TRAIN fold only,
    so there's no leakage). Price is log-transformed for training/prediction and
    converted back to USD for error reporting, since price is right-skewed."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ptype_cols = [c for c in X.columns if c.startswith("type_")]
    # recover the category label per row from the one-hot columns, for the baseline
    ptype = X[ptype_cols].idxmax(axis=1).str.replace("type_", "", regex=False).values

    mae_base, r2_base, mae_rf, r2_rf = [], [], [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ptype_train, ptype_test = ptype[train_idx], ptype[test_idx]

        # baseline: median price per product type, fit on train fold only
        train_medians = pd.Series(y_train).groupby(ptype_train).median()
        base_pred = pd.Series(ptype_test).map(train_medians).values
        mae_base.append(mean_absolute_error(y_test, base_pred))
        r2_base.append(r2_score(y_test, base_pred))

        rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42)
        rf.fit(X_train, np.log1p(y_train))
        pred = np.expm1(rf.predict(X_test))
        mae_rf.append(mean_absolute_error(y_test, pred))
        r2_rf.append(r2_score(y_test, pred))

    print(f"Baseline (category median) - MAE: ${np.mean(mae_base):,.0f}  R2: {np.mean(r2_base):.3f}")
    print(f"Random Forest              - MAE: ${np.mean(mae_rf):,.0f}  R2: {np.mean(r2_rf):.3f}")
    improvement = 100 * (1 - np.mean(mae_rf) / np.mean(mae_base))
    print(f"Random Forest reduces MAE by {improvement:.1f}% vs. the category-median baseline\n")


def plot_price_feature_importance(X: pd.DataFrame, y: np.ndarray, top_n: int = 12) -> None:
    """Fits a Random Forest on the full dataset (for interpretation, not evaluation
    - evaluate_price_model already reports honest out-of-sample performance) and
    plots the top features driving its price predictions."""
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42)
    rf.fit(X, np.log1p(y))
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(importances.index, importances.values, color="steelblue")
    plt.xlabel("Feature importance")
    plt.title("What Predicts Price? (Random Forest feature importance)")
    plt.tight_layout()
    plt.savefig(WORKING_DIRECTORY + "\\price_feature_importance.png", dpi=130)
    plt.show()
    print("Plot saved as price_feature_importance.png\n")

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

    # Fabric composition clustering
    fabric_X, material_names = build_material_matrix(df)
    print(f"List of materials used: {', '.join(sorted(material_names))}\n")
    plot_elbow(fabric_X)
    df = run_kmeans_clustering(df, fabric_X, n_clusters=N_CLUSTERS)
    visualize_clusters(df, fabric_X)
    df.drop(columns=["fabric_parsed"], errors="ignore").to_csv(WORKING_DIRECTORY + '\\' + CLUSTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"Clustered data saved to {WORKING_DIRECTORY + '\\' + CLUSTERED_CSV}")

    # Price tier clustering by product type
    # NOTE: this now reuses the same `df` from the fabric-clustering step above
    # (rather than re-reading the CSV into a separate df_prices) so that
    # PRODUCT_TYPE_NORMALIZED and PRICE_USD end up on the same DataFrame that
    # already has the fiber columns and premium_fiber_pct - build_price_model_dataset
    # below needs all of these present on one DataFrame at once.
    df["PRODUCT_TYPE_NORMALIZED"] = df["PRODUCT TYPE"].apply(normalize_product_type)
    df = clean_prices(df)
    df = cluster_prices_by_type(df)
    visualize_price_tiers(df)

    # Supervised price prediction
    price_X, price_y = build_price_model_dataset(df, material_names)
    print(f"Price model dataset: {price_X.shape[0]} products, {price_X.shape[1]} features\n")
    evaluate_price_model(price_X, price_y)
    plot_price_feature_importance(price_X, price_y)

    # # Save code as PDF
    # create_pdf()

if __name__ == "__main__":
    main()