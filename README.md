# Data_Visualization_Lab-9_Dashboard

For this lab we need to create a dashboard.
We may use any library such as R Shiny, Python Dash or D3 JS , We are using Python Dash.

This is a Python script that uses the Dash library to create an analytics dashboard for stroke prediction. The script first imports the necessary libraries such as pandas, dash, plotly, and datetime. It then loads the stroke prediction dataset from a CSV file, drops the 'id' column, and fills missing values in the 'bmi' column with the mean value.

The script then defines various functions to generate the dashboard components. These functions include a description card, a control card with dropdown menus for selecting gender, work type, heart disease, and hypertension, and two graphs - a heatmap and a scatter plot. The heatmap and scatter plot functions take in the selected values from the dropdown menus and filter the dataset accordingly before generating the plot.

The layout of the dashboard is defined using Dash's HTML and CSS components, which are organized into a column layout with a description card, control card, and two graphs.

Finally, the script defines two callback functions that take in the selected values from the dropdown menus and update the heatmap and scatter plot based on the filtered dataset.

The script can be run to launch the dashboard using the command python filename.py. The dashboard can then be accessed in a web browser at http://127.0.0.1:8050/. The user can select values from the dropdown menus to update the heatmap and scatter plot accordingly.

![](https://github.com/PoonamSuthar/Data_Visualization_Lab-9_Dashboard/blob/main/dashboard_GIF.gif)


