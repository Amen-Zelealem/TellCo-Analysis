import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


class HandsetVisualization:
    def __init__(self, custom_colors):
        self.custom_colors = custom_colors

    # Function to visualize top handsets side by side (DataFrame and chart)
    def visualize_top_handsets(self, top_handsets, top_n):
        st.write(f"### 📱 Top {top_n} Handsets")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the DataFrame in the first column
        with col1:
            st.write(top_handsets)

        # Display the bar chart in the second column
        with col2:
            fig, ax = plt.subplots()
            top_handsets.plot(kind="bar", ax=ax, color=self.custom_colors)
            ax.set_ylabel("Count")
            ax.set_title(f"Top {top_n} Handsets", fontsize=16)
            ax.grid(
                axis="y", linestyle="--", alpha=0.7
            )  # Add grid for better readability
            st.pyplot(fig)

    # Function to visualize top manufacturers side by side (DataFrame and chart)
    def visualize_top_manufacturers(self, top_manufacturers, top_n_manufacturers):
        st.write(f"### 🏭 Top {top_n_manufacturers} Manufacturers")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the DataFrame in the first column
        with col1:
            st.write(top_manufacturers)

        # Display the bar chart in the second column
        with col2:
            fig, ax = plt.subplots()
            top_manufacturers.plot(kind="bar", ax=ax, color=self.custom_colors)
            ax.set_ylabel("Count")
            ax.set_title(f"Top {top_n_manufacturers} Manufacturers", fontsize=16)
            plt.xticks(rotation=90)  # Rotate x labels for better readability
            ax.grid(
                axis="y", linestyle="--", alpha=0.7
            )  # Add grid for better readability
            st.pyplot(fig)

    # Function to visualize top handsets for each manufacturer (DataFrame and chart)
    def visualize_top_handsets_per_manufacturer(
        self, top_handsets_per_manufacturer, manufacturers, top_n_handsets
    ):
        for manufacturer, handsets in top_handsets_per_manufacturer.items():
            st.write(f"### 📊 Top {top_n_handsets} Handsets for {manufacturer}")

            # Create two columns for the manufacturer-specific results
            col1, col2 = st.columns(2)

            # Display the DataFrame in the first column
            with col1:
                st.write(handsets)

            # Display the bar chart in the second column
            with col2:
                fig, ax = plt.subplots()

                # Generate a color palette based on the number of handsets
                colors = sns.color_palette("Set2", len(handsets))  # Set color palette

                # Plot the bar chart with individual bar colors
                handsets.plot(
                    kind="bar", ax=ax, color=self.custom_colors, edgecolor="black"
                )  # Use edgecolor for contrast

                ax.set_ylabel("Count")
                ax.set_title(
                    f"Top {top_n_handsets} Handsets for {manufacturer}", fontsize=16
                )
                plt.xticks(rotation=90)  # Rotate x labels for better readability
                ax.grid(
                    axis="y", linestyle="--", alpha=0.7
                )  # Add grid for better readability
                st.pyplot(fig)
