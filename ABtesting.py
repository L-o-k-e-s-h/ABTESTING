import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import streamlit as st

# Import data from ABTest sheet
abtest = pd.read_excel("C:\\Users\\lokes\\Downloads\\AssignmentData.xlsx", sheet_name='ABTest')
# Perform exploratory analysis
st.title("Exploratory Analysis")

st.write(abtest)
st.write(abtest.info())

# Time-series visualization
st.title("Time-series visualization")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=abtest, x='Date', y='Clicks', hue='Device', ax=ax)
ax.set_title('Total Number of Clicks by Device Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Total Number of Clicks')
ax.legend(title='Device')
ax.grid(True)  # Add gridlines for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
st.pyplot(fig)


# Calculate required sample size
def calculate_sample_size(mde, alpha, power):
    # Use statistical methods to calculate sample size
    # For example, you can use power analysis or online calculators
    sample_size = 100  # Placeholder value
    return sample_size

# Example: Calculate sample size
mde = 0.03  # Minimum Detectable Effect (3%)
alpha = 0.05  # Significance Level (α = 95%)
power = 0.8  # Statistical Power (1-β = 80%)
sample_size = calculate_sample_size(mde, alpha, power)
st.write("Required Sample Size:", sample_size)

# Hypothesis testing function
def perform_ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level):
    # Perform data validation
    if control_visitors < control_conversions or treatment_visitors < treatment_conversions:
        return "Invalid input: Visitors should be greater than or equal to conversions."

    # Perform t-test
    _, p_value = ttest_ind([1] * control_conversions + [0] * (control_visitors - control_conversions),
                           [1] * treatment_conversions + [0] * (treatment_visitors - treatment_conversions))
    
    # Determine significance based on confidence level
    alpha = 1 - (confidence_level / 100)
    if p_value < alpha:
        return "Experiment Group is Better"
    elif p_value > 1 - alpha:
        return "Control Group is Better"
    else:
        return "Indeterminate"

# Streamlit app
def main():
    st.title('A/B Test Hypothesis Testing App')
    
    st.write('Enter the following inputs to perform the hypothesis test:')
    
    control_visitors = st.number_input('Control Group Visitors', min_value=0)
    control_conversions = st.number_input('Control Group Conversions', min_value=0)
    treatment_visitors = st.number_input('Treatment Group Visitors', min_value=0)
    treatment_conversions = st.number_input('Treatment Group Conversions', min_value=0)
    confidence_level = st.radio('Confidence Level', [90, 95, 99])
    
    if st.button('Perform Hypothesis Test'):
        result = perform_ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level)
        st.write('Result of A/B Test:', result)

if __name__ == '__main__':
    main()
