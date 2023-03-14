PYTHON          := python3.9

SRC             := src
PYCACHE         := $(SRC)/__pycache__

DATA_DIR        := training_samples
SAMPLES         := $(shell find $(DATA_DIR) -type f)
MODEL           := catifer.sklearn

MAIN            := $(SRC)/live.py
MODEL_GEN       := $(SRC)/model_gen.py
CAT_CLASS       := $(SRC)/cat_sample.py

all: gen_model

$(MODEL): $(SAMPLES) $(MODEL_GEN) $(CAT_CLASS)
	$(PYTHON) $(MODEL_GEN) -o $(MODEL) -s $(DATA_DIR)

# Alias for model.
gen_model: $(MODEL)

.PHONY: run clean

run: $(MODEL) $(MAIN)
	$(PYTHON) $(MAIN) -m $(MODEL)

clean:
	rm -rf $(PYCACHE) $(MODEL)
