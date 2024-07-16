# Directory containing LAZ files
LAZ_DIR = data/laz

# Directory to save NPZ files
NPZ_DIR = data/npz

# Python script to run
SCRIPT = laz_to_xyz.py

# Find all .laz files in the LAZ_DIR
LAZ_FILES = $(wildcard $(LAZ_DIR)/*.laz)

# Define the corresponding .npz files
NPZ_FILES = $(LAZ_FILES:$(LAZ_DIR)/%.laz=$(NPZ_DIR)/%.npz)

# Default target: Generate all NPZ files
all: $(NPZ_FILES)

# Rule to convert .laz to .npz
$(NPZ_DIR)/%.npz: $(LAZ_DIR)/%.laz
	python $(SCRIPT) --input $< --output $@

# Clean target: Remove all generated .npz files
clean:
	rm -f $(NPZ_DIR)/*.npz

.PHONY: all clean
