
import streamlit as st
from combined_code import load_metadata, remove_from_metadata

def main():
    st.title("🖼️ Image Management Dashboard")
    st.sidebar.header("⚙️ Options")

    # Load Metadata
    try:
        metadata = load_metadata()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Display Images
    st.sidebar.write(f"Total Images in Metadata: {len(metadata)}")
    selected_image = None

    for image_path in metadata:
        # Display image
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(image_path, caption=metadata[image_path]["name"], use_column_width=True)
        with col2:
            # Provide an option to remove the image
            if st.button(f"Remove {metadata[image_path]['name']}", key=image_path):
                selected_image = image_path

    # Process Image Removal
    if selected_image:
        remove_from_metadata(selected_image)
        st.success(f"Removed {selected_image} from metadata. Refresh the page to see changes.")

if __name__ == "__main__":
    main()

