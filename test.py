import pdf_extraction


if __name__ == "__main__":
    pdf_path = r"C:\Users\kriti\OneDrive\Pictures/physics_test1.pdf"  # Or your actual path
    output_file_path = "final_output.txt"
    text, success = process_pdf_with_timeout_and_output(pdf_path, output_file_path)
        # text = extract_text_from_file(file_path)
    if not success:
         print("Could not extract text.")