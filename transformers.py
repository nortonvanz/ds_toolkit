class RenameDfColumns(BaseEstimator, TransformerMixin):
    """Transform DataFrame column names to snake_case format.
    
    A scikit-learn transformer that converts all column names in a pandas DataFrame to snake_case naming convention using the inflection library.
        For example: 'FirstName' becomes 'first_name', 'DOB' becomes 'dob'.
    
    Attributes
    ----------
    None
    
    Methods
    -------
    fit(X, y=None)
        No-op method that returns self (required for scikit-learn API).
    transform(X, y=None)
        Converts DataFrame column names to snake_case.
    """
    def fit(self, X, y=None):
        """No-op method required for scikit-learn transformer API.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input DataFrame, not used
        y : None
            Ignored
            
        Returns
        -------
        self : object
            Returns self
        """
        return self
    
    def transform(self, X, y=None):
        """Transform DataFrame column names to snake_case.
        
        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame whose columns need to be renamed
        y : None
            Ignored
            
        Returns
        -------
        pandas.DataFrame
            A copy of input DataFrame with renamed columns
        """
        return X.copy().rename(columns=lambda x: inflection.underscore(x))
    


class FilterFeatures(BaseEstimator, TransformerMixin):
    """Remove specified columns from a DataFrame.
    
    A scikit-learn transformer that removes specified columns from a pandas DataFrame.
    If no columns are specified during initialization, no columns will be removed.
    
    Parameters
    ----------
    columns : list or None, default=None
        List of column names to remove from the DataFrame.
        If None or empty list, no columns will be removed.
    
    Attributes
    ----------
    columns : list
        List of columns to be removed. Empty list if no columns specified.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    >>> filter_features = FilterFeatures(columns=['A', 'B'])
    >>> df_filtered = filter_features.transform(df)
    >>> df_filtered.columns
    Index(['C'], dtype='object')
    """
    
    def __init__(self, columns=None):
        """Initialize the transformer with columns to remove.
        
        Parameters
        ----------
        columns : list or None, default=None
            List of column names to remove from the DataFrame.
            If None, an empty list will be used.
        """
        self.columns = columns if columns else []
    
    def fit(self, X, y=None):
        """No-op method required for scikit-learn transformer API.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input DataFrame, not used
        y : None
            Ignored
            
        Returns
        -------
        self : object
            Returns self
        """
        return self
    
    def transform(self, X, y=None):
        """Remove specified columns from the input DataFrame.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input DataFrame from which columns will be removed
        y : None
            Ignored
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with specified columns removed
            
        Raises
        ------
        KeyError
            If any of the specified columns are not present in the input DataFrame
        """
        return X.drop(columns=self.columns)