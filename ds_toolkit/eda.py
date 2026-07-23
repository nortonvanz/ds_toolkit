# Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker as mtick; 
from matplotlib.ticker import ScalarFormatter
from scipy.stats                 import chi2_contingency
sns.set_palette("deep") # Design


# Cleaning
def inspect_outliers(df, numeric_columns):
    """Visualize and analyze outliers in numeric features using box plots and IQR method.
    
    Creates a 4-column grid of box plots for numeric features, with outlier statistics displayed below each plot. Uses the Interquartile Range (IQR) method for outlier detection. 
    The visualization includes the original data scale and prevents scientific notation for better readability.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the numeric features to analyze numeric_columns : list
        List of column names containing numeric data to analyze for outliers
        
    Returns
    -------
    None
        Displays a matplotlib figure with:
        - Grid of box plots (4 columns)
        - Outlier statistics below each plot
        - Original data scale (no scientific notation)
        
    Notes
    -----
    Visualization details:
    - Box plots arranged in a 4-column grid
    - White background text box for statistics
    - Increased vertical spacing between plots (h_pad=3)
    - Empty subplots are removed if number of features isn't divisible by 4
    
    For each feature, displays:
    - Box plot showing distribution and outliers
    - Number and percentage of outliers
    - Lower and upper bounds for outlier detection
    
    The IQR method defines outliers as:
    - Lower outliers: < Q1 - 1.5 * IQR
    - Upper outliers: > Q3 + 1.5 * IQR
    where:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    
    Examples
    --------
    >>> numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    >>> inspect_outliers(df, numeric_cols)
    """
    
    # Create subplots for each numeric column
    n_cols = 4
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()
    
    # Plot data
    for idx, column in enumerate(numeric_columns):

        # Calculate outlier statistics using IQR method
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        
        # Create box plot
        sns.boxplot(data=df, y=column, ax=axes[idx])
        axes[idx].set_title(f'Box Plot - {column}')
        
        # Format y-axis to prevent scientific notation
        axes[idx].yaxis.set_major_formatter(ScalarFormatter())
        axes[idx].ticklabel_format(style='plain', axis='y')
        
        # Add statistics text below plot
        stats_text = f'Outliers: {len(outliers)} ({(len(outliers)/len(df))*100:.1f}%)\nBounds: [{lower_bound:,.1f} | {upper_bound:,.1f}]'
        axes[idx].text(0.5, -0.12, stats_text, 
                      horizontalalignment='center',
                      transform=axes[idx].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
     # Remove empty subplots
    for idx in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout(h_pad=3)
    plt.show()


# Univariate Analysis
def statistics(data):
    """Calculate comprehensive descriptive statistics for numerical data.
    
    Computes various statistical metrics including central tendency, dispersion, and distribution shape for all numeric columns in the input DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing numerical columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with the following statistics for each numeric column:
        - non-null : int
            Count of non-null values
        - range : float
            Difference between maximum and minimum values
        - min : float
            Minimum value
        - quant25 : float
            First quartile (25th percentile)
        - median : float
            Median (50th percentile)
        - quant75 : float
            Third quartile (75th percentile)
        - max : float
            Maximum value
        - mean : float
            Arithmetic mean
        - std : float
            Standard deviation
        - skew : float
            Skewness (measure of distribution asymmetry)
        - kurtosis : float
            Kurtosis (measure of distribution tailedness)
            
    Notes
    -----
    - All metrics are rounded to 1 decimal place
    - Only columns with int or float datatypes are analyzed
    - NaN values are automatically excluded from calculations
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2.5, 3.5, 4.5, 5.5, 6.5],
    ...     'C': ['a', 'b', 'c', 'd', 'e']
    ... })
    >>> stats = statistics(df)
    >>> print(stats.columns)
    Index(['non-null', 'range', 'min', 'quant25', 'median', 'quant75', 'max', 'mean', 'std', 'skew', 'kurtosis'], dtype='object')
    """
    # Select only numeric columns
    num_data = data.select_dtypes(include=['int', 'float'])

    # Central Tendency
    mean = num_data.apply(np.mean)

    # Quantiles
    q25 = num_data.quantile(0.25)
    q50 = num_data.quantile(0.5)
    q75 = num_data.quantile(0.75)
    range_ = num_data.apply(lambda x: x.max() - x.min())
    count = num_data.count()

    # Dispersion
    min_ = num_data.apply(min)
    max_ = num_data.apply(max)
    std = num_data.apply(np.std)

    # Distribution Shape
    skew = num_data.apply(lambda x: x.skew())
    kurtosis = num_data.apply(lambda x: x.kurtosis())

    metrics = pd.DataFrame({
        'non-null': count,
        'range': range_,
        'min': min_,
        'quant25': q25,
        'median': q50,
        'quant75': q75,
        'max': max_,
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis
    })
    
    return np.round(metrics, 1)

def plot_target_distribution(df, target_col, figsize=(10, 6)):
    """Plot distribution of target variable with percentage labels.
    
    Creates a count plot showing the distribution of target variable classes,
    with percentage labels on top of each bar.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the target variable
    target_col : str
        Name of the target variable column
    figsize : tuple, default=(10, 6)
        Figure size as (width, height)
        
    Returns
    -------
    None
        Displays a matplotlib figure with:
        - Count plot of target variable distribution
        - Percentage labels on top of each bar
        
    Notes
    -----
    Visualization details:
    - Bar plot showing class frequencies
    - Percentage labels positioned above each bar
    - Title includes target variable name
    
    Examples
    --------
    >>> plot_target_distribution(train, 'exited')
    >>> plot_target_distribution(df, 'churn', figsize=(12, 8))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create count plot
    sns.countplot(data=df, x=target_col, ax=ax)
    
    # Set title
    ax.set_title(f"Distribution of {target_col} (target)", fontsize=12)
    # Set x and y labels names
    ax.set_xlabel(target_col, fontsize=10)
    ax.set_ylabel("Total Volume", fontsize=10) 
    
    # Add percentage labels on top of bars
    total = len(df)
    for p in ax.patches:
        percentage = p.get_height()/total * 100
        ax.text(
            p.get_x() + p.get_width()/2.,
            p.get_height() + total*0.01,  # Adjust label position based on data size
            '{:.2f}%'.format(percentage),
            ha='center'
        )
    
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df, figsize=(20, 4)):
    """Plot distributions of numerical features with histograms and KDE.
    
    Creates a grid of distribution plots for numerical features, showing both histogram and kernel density estimation (KDE) for better visualization of the data distribution shape.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing only numerical features to analyze
    figsize : tuple, default=(20, 4)
        Base figure size for the plot. Height will be adjusted based on number of features
        
    Returns
    -------
    None
        Displays a matplotlib figure with:
        - Grid of distribution plots (3 columns)
        - Histogram and KDE for each feature
        - Automatic adjustment for number of features
        
    Notes
    -----
    Visualization details:
    - 3-column grid layout
    - Histogram with KDE overlay
    - Empty subplots are removed if number of features isn't divisible by 3
    - Automatic subplot sizing based on number of features
    
    Examples
    --------
    >>> train_num = train.select_dtypes(include=['int64', 'float64'])
    >>> plot_numerical_distributions(train_num)
    """
    num_features = df.columns
    n_features = len(num_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Adjust figure height based on number of rows
    fig_height = figsize[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height))
    axes = axes.flatten()

    for idx, feature in enumerate(num_features):
        # Create histogram with KDE
        sns.histplot(df[feature], kde=True, ax=axes[idx], bins=80)
        axes[idx].set_title(f'Distribution of {feature}')
        
        # Format axis labels
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Total Volume')

    # Remove empty subplots
    for idx in range(len(num_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df, figsize=(12, 4), max_categories=15):
    """Plot distributions of categorical features using count plots.
    
    Creates a grid of count plots for categorical features, showing the frequency
    of each category in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing only categorical features to analyze
    figsize : tuple, default=(12, 4)
        Base figure size for the plot. Height will be adjusted based on
        number of features
    max_categories : int, default=15
        Maximum number of categories to display per feature. If a feature has more
        categories, only the top ones will be shown.
        
    Returns
    -------
    None
        Displays a matplotlib figure with:
        - Grid of count plots (3 columns)
        - Frequency counts for each category
        - Automatic adjustment for number of features
        
    Notes
    -----
    Visualization details:
    - 3-column grid layout
    - Vertical bar plots showing category counts
    - Category labels rotated 45 degrees for better readability
    - Empty subplots are removed if number of features isn't divisible by 3
    - Automatic subplot sizing based on number of features
    - Uses pre-computed value counts for faster plotting with large datasets
    
    Examples
    --------
    >>> train_cat = train.select_dtypes(exclude=['int64', 'float64'])
    >>> plot_categorical_distributions(train_cat)
    """
    cat_features = df.columns
    n_features = len(cat_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Adjust figure height based on number of rows
    fig_height = figsize[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height))
    
    # Handle axis flattening based on dimensions
    if n_rows == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()

    for idx, feature in enumerate(cat_features):
        # Pre-compute value counts for performance
        value_counts = df[feature].value_counts()
        
        # Limit to top categories if there are too many
        if len(value_counts) > max_categories:
            top_counts = value_counts.nlargest(max_categories)
            other_count = value_counts.iloc[max_categories:].sum()
            value_counts = pd.concat([top_counts, pd.Series({'Other': other_count})])
        
        # Plot with matplotlib
        axes[idx].bar(value_counts.index.astype(str), value_counts.values)
        axes[idx].set_title(f'Distribution of {feature}')
        
        # Rotate and format axis labels
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Total Volume')
        
        # Ensure x-tick labels are visible and not overlapping
        plt.setp(axes[idx].xaxis.get_majorticklabels(), ha='right')

    # Remove empty subplots
    for idx in range(len(cat_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# Multivariate Analysis
def plot_numerical_correlations(df, figsize=(10, 5)):
    """Plot Pearson correlation matrix for numerical variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical variables
    figsize : tuple, default=(10, 5)
        Figure size as (width, height)
        
    Returns
    -------
    None
        Displays a heatmap showing Pearson correlations
        
    Notes
    -----
    Interpretation of Pearson's correlation:
    - -1: Perfect negative correlation
    - 0: No correlation
    - +1: Perfect positive correlation
    
    Correlation strength:
    - ±0.00 to ±0.19: Very weak
    - ±0.20 to ±0.39: Weak
    - ±0.40 to ±0.59: Moderate
    - ±0.60 to ±0.79: Strong
    - ±0.80 to ±1.00: Very strong
    
    Examples
    --------
    >>> train_num = train.select_dtypes(include=['int64', 'float64'])
    >>> plot_numerical_correlations(train_num)
    """
    # Calculate correlation matrix
    correlation_matrix = df.corr(method='pearson')
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f',
                center=0)
    plt.title('Pearson Correlation Matrix')
    plt.show()

def plot_categorical_correlations(df, figsize=(10, 5), max_unique_values=50):
    """Plot Cramér's V correlations between categorical variables as a heatmap.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing categorical variables and target
    figsize : tuple, default=(10, 5)
        Figure size as (width, height)
    max_unique_values : int, default=50
        Maximum number of unique values allowed in a column to prevent memory crashes.
    """
    # 1. Filter out high-cardinality or continuous columns
    valid_cols = []
    for col in df.columns:
        num_unique = df[col].nunique()
        if num_unique > max_unique_values:
            print(f"Skipping '{col}': Too many unique values ({num_unique}). Might be continuous or high-cardinality.")
        elif num_unique <= 1:
            print(f"Skipping '{col}': Only has {num_unique} unique value (no variance).")
        else:
            valid_cols.append(col)
            
    if len(valid_cols) < 2:
        print("Error: Not enough valid categorical columns left to compute correlations.")
        return
        
    working_df = df[valid_cols]

    def cramers_v(x, y):
        # Clean pairs by dropping rows with NaNs in either column
        mask = x.notna() & y.notna()
        if not mask.any():
            return np.nan
        
        # Fast contingency table approach using pandas groupby
        confusion_matrix = pd.concat([x[mask], y[mask]], axis=1).groupby([x.name, y.name]).size().unstack(fill_value=0)
        
        if confusion_matrix.size == 0:
            return 0.0
            
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.values.sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0:
            return 0.0
            
        return np.sqrt(chi2 / (n * min_dim))
    
    # 2. Build the correlation matrix safely
    n_cols = len(valid_cols)
    corr_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:  # Matrix is symmetric, save computation time
                val = cramers_v(working_df.iloc[:, i], working_df.iloc[:, j])
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val

    # Convert to DataFrame for easier plotting
    corr_df = pd.DataFrame(corr_matrix, index=valid_cols, columns=valid_cols)
    
    # 3. Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                vmin=0,
                vmax=1,
                center=0.5,
                square=True,
                cbar_kws={'label': "Cramér's V Correlation"})
    
    plt.title("Cramér's V Correlation Matrix")
    plt.tight_layout()
    plt.show()


# Bivariate Analysis
def test_statistical_significance(df, feature, value, target):
    """
    Evaluates whether the difference in target (event) rates between specific feature group(s)
    and all other groups combined is statistically significant.

    This function uses a Chi-Square Test of Independence on a 2x2 contingency table to determine
    if the observed deviation is likely due to random chance or a true underlying pattern, which
    is especially useful for verifying low-volume segments.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing both the feature and the target columns.
    feature : str
        The name of the categorical or discrete column to evaluate (e.g., 'Balance Green Zone').
    value : str, int, or list
        The specific attribute value, or a list of values, within the feature column being targeted
        for the hypothesis test. Each value is tested independently against all other groups combined.
        Examples: 'HIBRIDA', 9, or [9, 10].
    target : str
        The name of the binary target column (containing 0 and 1, or False and True) representing
        the outcome event (e.g., 'Exited' or 'Is Fraud').

    Returns:
    --------
    None
        Prints a formatted summary of volumes, target event rates, the calculated
        p-value, and a final significance conclusion for each value tested.
    """
    # Normalize value to always be a list for uniform processing
    values = value if isinstance(value, list) else [value]

    for val in values:
        # 1. Create a binary indicator: Is it the target value or 'Others'?
        is_target_value = (df[feature].astype(str) == str(val))

        # 2. Extract total counts and target (positive event) counts
        total_val = is_target_value.sum()
        pos_val = df.loc[is_target_value, target].sum()
        neg_val = total_val - pos_val

        # All other groups combined
        total_others = (~is_target_value).sum()
        pos_others = df.loc[~is_target_value, target].sum()
        neg_others = total_others - pos_others

        # 3. Handle Edge Case: Avoid running test if counts are zero
        if total_val == 0 or total_others == 0:
            print(f"Cannot perform test. Group '{val}' has 0 records or contains the entire dataset.")
            if len(values) > 1:
                print()
            continue

        # 4. Build the 2x2 contingency table for the Chi-Square test
        contingency_table = [
            [pos_val, neg_val],
            [pos_others, neg_others]
        ]

        # 5. Run Chi-Square Test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # 6. Calculate Rates for cleaner output
        rate_val = (pos_val / total_val) * 100 if total_val > 0 else 0
        rate_others = (pos_others / total_others) * 100 if total_others > 0 else 0

        # Print formatted output
        print(f"=== Significance Analysis for {feature} == '{val}' ===")
        print(f"• Target Group Volume: {total_val} | Event Rate: {rate_val:.2f}%")
        print(f"• Others Group Volume: {total_others} | Event Rate: {rate_others:.2f}%")
        print(f"• P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Statistically Significant: The difference is highly unlikely to be random chance.")
            if total_val < 100:
                print("Note: Although statistically significant, the absolute volume is very low (<100). Proceed with caution.")
        else:
            print("Not Significant: The sample size is too low or the difference is too small to be trusted.")

        if len(values) > 1:
            print()

def plot_comprehensive_target_distribution(
    df,
    feature,
    target,
    bin_size=5,
    figsize=(12, 8),
    title=None,
    abs_decimals=1,
    rate_decimals=2,
):
    """Plots a two-row distribution dashboard for analyzing a feature against a

    target variable.

    The top plot visualizes the absolute sum of the target across different
    feature groups along with its concentration percentage. The bottom plot
    displays a dual-axis chart mapping total record volumes (bars) against the
    average target rate (line).

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing both the feature and target columns.
    feature : str
        The name of the column to analyze. Automatically handles logic for:
          - Numeric with <= 10 unique values (treated as Discrete/Binary).
          - Numeric with > 10 unique values (binned into intervals using
          `bin_size`).
          - Non-numeric columns (treated as Categorical).
    target : str
        The name of the binary or numeric target variable to aggregate (sum,
        count, mean).
    bin_size : int, default 50
        The step size used to generate bins/intervals if `feature` is continuous
        numerical.
    figsize : tuple of (float, float), default (9, 9)
        Width and height of the matplotlib figure in inches.
    title : str, optional
        Custom title for the entire dashboard. If None, a title is automatically
        generated using the feature name.
    abs_decimals : int, default 0
        Number of decimal places to show for the percentage concentration text
        labels in the top chart.
    rate_decimals : int, default 0
        Number of decimal places to show for the target rate labels and Y-axis
        ticks in the bottom-right line chart.

    Returns:
    --------
    None
        Displays the generated matplotlib dashboard directly using `plt.show()`.
    """
    # Create subplots: 2 rows, 1 column
    fig, (ax_top, ax_bottom_left) = plt.subplots(
        nrows=2, ncols=1, figsize=figsize
    )

    df_temp = df[[feature, target]].copy()
    group_col = f"{feature}_group"

    # Check uniqueness threshold to identify binary / low-range discrete variables
    is_numeric = pd.api.types.is_numeric_dtype(df[feature])
    unique_count = df[feature].nunique()

    # 1. Handle Feature Type (Discrete/Binary vs. Continuous Numerical vs. Categorical)
    if is_numeric and unique_count <= 10:
        df_temp[group_col] = df_temp[feature].astype(str)
        x_label = f"{feature.replace('_', ' ').title()} (Discrete Values)"
        sort_by_value = True

    elif is_numeric:
        min_val = int(df[feature].min())
        max_val = int(df[feature].max())

        start_bin = (min_val // bin_size) * bin_size
        end_bin = ((max_val // bin_size) + 1) * bin_size + 1
        calculated_bins = np.arange(start_bin, end_bin, bin_size)

        bin_labels = [
            f"{calculated_bins[i]} - {calculated_bins[i+1]}"
            for i in range(len(calculated_bins) - 1)
        ]

        df_temp[group_col] = pd.cut(
            df_temp[feature],
            bins=calculated_bins,
            labels=bin_labels,
            include_lowest=True,
        )
        x_label = f"{feature.replace('_', ' ').title()} Intervals"
        sort_by_value = False
    else:
        df_temp[group_col] = df_temp[feature].astype(str)
        x_label = f"{feature.replace('_', ' ').title()}"
        sort_by_value = False

    # 2. Aggregation: Get Sum, Count, and Mean
    df_grouped = (
        df_temp.groupby(group_col, observed=False)
        .agg(
            target_sum=(target, "sum"),
            total_count=(target, "count"),
            target_rate=(target, "mean"),
        )
        .reset_index()
    )

    # FILTER BINS: Keep only rows with active data
    df_grouped = df_grouped[df_grouped["total_count"] > 0].reset_index(
        drop=True
    )

    if isinstance(df_grouped[group_col].dtype, pd.CategoricalDtype):
        df_grouped[group_col] = (
            df_grouped[group_col].cat.remove_unused_categories()
        )

    # Convert to string to prevent Seaborn's categorical retention memory trap
    df_grouped[group_col] = df_grouped[group_col].astype(str)

    # Ordering logic
    if sort_by_value:
        df_grouped = df_grouped.iloc[
            np.argsort(pd.to_numeric(df_grouped[group_col], errors="coerce"))
        ]
    elif not is_numeric:
        df_grouped = df_grouped.sort_values(by="total_count", ascending=False)

    # Calculate overall target sum across all active groups for top chart concentration metrics
    total_target_sum = df_grouped["target_sum"].sum()

    # =========================================================================
    # PLOT 1: ABSOLUTE TARGET COUNT WITH % CONCENTRATION (TOP)
    # =========================================================================
    sns.barplot(
        x=group_col, y="target_sum", data=df_grouped, color="#CC2B0E", ax=ax_top
    )
    ax_top.set_title(
        f"Absolute Total of {target.replace('_', ' ').title()} by {feature.replace('_', ' ').title()}",
        fontsize=10,
        fontweight="bold",
    )
    ax_top.set_ylabel(
        f"Total {target.replace('_', ' ').title()}",
        color="#CC2B0E",
        fontweight="bold",
    )
    ax_top.tick_params(axis="y", labelcolor="#CC2B0E")
    ax_top.set_xlabel("")

    # Set the labels below the top bars to size 8
    ax_top.tick_params(axis="x", rotation=25, labelrotation=25, labelsize=8)
    for tick in ax_top.get_xticklabels():
        tick.set_ha("right")

    # Dynamic headroom for Top Chart
    max_target_sum = df_grouped["target_sum"].max()
    if max_target_sum > 0:
        ax_top.set_ylim(0, max_target_sum * 1.15)  # 15% headroom padding

    for p in ax_top.patches:
        height = p.get_height()
        if height > 0:
            # Calculate concentration percentage relative to total targets found
            pct_share = (
                (height / total_target_sum) * 100 if total_target_sum > 0 else 0.0
            )
            ax_top.annotate(
                f"{int(height)} ({pct_share:.{abs_decimals}f}%)",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#CC2B0E",
                xytext=(0, 4),
                textcoords="offset points",
            )

    # =========================================================================
    # PLOT 2: DUAL AXIS VOLUME VS RATE (BOTTOM)
    # =========================================================================
    # Left Axis - Volume (Bars)
    sns.barplot(
        x=group_col,
        y="total_count",
        data=df_grouped,
        color="#126801FF",
        alpha=0.6,
        ax=ax_bottom_left,
    )
    ax_bottom_left.set_title(
        f"Percentage Rate of {target.replace('_', ' ').title()} vs {feature.replace('_', ' ').title()}",
        fontsize=10,
        fontweight="bold",
    )
    ax_bottom_left.set_ylabel(
        "Total Volume", color="#126801FF", fontweight="bold"
    )
    ax_bottom_left.set_xlabel(x_label)
    ax_bottom_left.tick_params(axis="y", labelcolor="#126801FF")

    # Set the labels below the bottom bars to size 8
    ax_bottom_left.tick_params(
        axis="x", rotation=30, labelrotation=30, labelsize=8
    )
    for tick in ax_bottom_left.get_xticklabels():
        tick.set_ha("right")

    # Dynamic headroom for Volume Bars (Bottom Left)
    max_total_count = df_grouped["total_count"].max()
    if max_total_count > 0:
        ax_bottom_left.set_ylim(
            0, max_total_count * 1.15
        )  # 15% headroom padding

    for p in ax_bottom_left.patches:
        height = p.get_height()
        if height > 0:
            ax_bottom_left.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#126801FF",
                xytext=(0, 4), #xytext=(0, 8),
                textcoords="offset points",
            )

    # Right Axis - Rate (Line)
    ax_bottom_right = ax_bottom_left.twinx()
    sns.lineplot(
        x=df_grouped[group_col],
        y=df_grouped["target_rate"] * 100,
        color="#CC2B0E",
        marker="o",
        linewidth=0.5,
        ax=ax_bottom_right,
    )
    ax_bottom_right.set_ylabel(
        f"{target.replace('_', ' ').title()} Rate (%)",
        color="#CC2B0E",
        fontweight="bold",
    )
    ax_bottom_right.tick_params(axis="y", labelcolor="#CC2B0E")

    # Format right y-axis tick values to match user-defined rate decimals
    ax_bottom_right.yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=100, decimals=rate_decimals)
    )

    # Dynamic headroom for Rate Line (Bottom Right)
    max_target_rate = (df_grouped["target_rate"] * 100).max()
    if max_target_rate > 0:
        ax_bottom_right.set_ylim(
            0, max_target_rate * 1.20
        )  # 20% headroom padding

    # Add percentage labels below Line markers with dynamic decimal places
    for x_idx, y_val in enumerate(df_grouped["target_rate"] * 100):
        if not np.isnan(y_val):
            ax_bottom_right.annotate(
                f"{y_val:.{rate_decimals}f}%",
                (x_idx, y_val),
                ha="center",
                va="top",
                fontsize=8,
                color="#CC2B0E",
                fontweight="bold",
                xytext=(4, 13), #xytext=(4, 11),
                textcoords="offset points",
            )

    # =========================================================================
    # Global Layout Tweaks (Tighter Padding)
    # =========================================================================
    main_title = (
        title
        if title
        else f"Target Distribution Analysis for: {feature.replace('_', ' ').title()}"
    )
    fig.suptitle(main_title, fontsize=11, fontweight="bold", y=0.96)

    plt.subplots_adjust(
        top=0.91, bottom=0.11, hspace=0.30, left=0.10, right=0.90
    )
    plt.show()