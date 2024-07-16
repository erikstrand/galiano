# Directory containing LAZ files
LAZ_DIR = data/laz

# Directory to save NPZ files
NPZ_DIR = data/npz

# Python script to run
XYZ_SCRIPT = laz_to_xyz.py
CLASS_SCRIPT = laz_to_classification.py

# Find all .laz files in the LAZ_DIR
LAZ_FILES = $(wildcard $(LAZ_DIR)/*.laz)

# Define the corresponding .npz files
NPZ_FILES_XYZ = $(LAZ_FILES:$(LAZ_DIR)/%.laz=$(NPZ_DIR)/%_xyz.npz)
NPZ_FILES_CLASS = $(LAZ_FILES:$(LAZ_DIR)/%.laz=$(NPZ_DIR)/%_class.npz)

# Default target: Generate all NPZ files
all: $(NPZ_FILES_XYZ) $(NPZ_FILES_CLASS)

# Rule to convert .laz to .npz for XYZ
$(NPZ_DIR)/%_xyz.npz: $(LAZ_DIR)/%.laz
	python $(SCRIPT) --input $< --output $@

# Rule to convert .laz to .npz for classification
$(NPZ_DIR)/%_class.npz: $(LAZ_DIR)/%.laz
	python $(CLASS_SCRIPT) --input $< --output $@

# Clean target: Remove all generated .npz files
clean:
	rm -f $(NPZ_DIR)/*.npz

.PHONY: all clean
