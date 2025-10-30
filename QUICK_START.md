# Quick Start Guide - Updated File Structure

## ✅ All Scripts Now Work with New Structure!

The project has been reorganized with a centralized configuration system. All paths are automatically handled.

## Running Scripts

### From Project Root (Recommended)

**1. Run Tournament Simulation:**
```bash
python run_simulation.py
```

**2. Process New XML Files:**
```bash
cd scripts
python batch_processor.py
```

**3. Train Models:**
```bash
cd scripts
python train_xgboost_with_players.py
```

**4. Validate Data:**
```bash
cd scripts
python validate_data.py
```

### From Scripts Directory

All scripts can also be run from within the `scripts/` directory:

```bash
cd scripts
python simulate_remaining_matches.py
python parse_volleyball_data.py
python feature_engineering_with_players.py
```

## Key Updates

### Configuration System
- All file paths are centralized in `scripts/config.py`
- Scripts automatically find files in correct directories
- No need to manually specify paths

### Updated Scripts
The following scripts have been updated to use the new structure:
- ✅ `simulate_remaining_matches.py` - Tournament simulation
- ✅ `batch_processor.py` - Batch processing pipeline
- ✅ `parse_volleyball_data.py` - XML parsing
- ✅ `config.py` - NEW: Centralized configuration

### File Locations
Scripts now automatically look for files in:
- **XML files**: `data/xml_files/`
- **CSV files**: `data/csv_files/`
- **Database**: `data/databases/volleyball_data.db`
- **Models**: `models/best_model_with_players.pkl`
- **Outputs**: `outputs/`

## Adding New Match Data

1. **Download XML files** and place in `data/xml_files/`:
   ```bash
   curl -o data/xml_files/new_match.xml 'URL'
   ```

2. **Process the data**:
   ```bash
   cd scripts
   python batch_processor.py
   ```

3. **Retrain models** (optional):
   ```bash
   python train_xgboost_with_players.py
   ```

4. **Run simulation**:
   ```bash
   cd ..
   python run_simulation.py
   ```

## Checking Configuration

View current configuration and file status:
```bash
python scripts/config.py
```

This shows:
- All directory paths
- Which files exist
- Count of XML files
- Database status

## Troubleshooting

### "Module not found" errors
Make sure you're in the correct directory:
- For `run_simulation.py`: run from project root
- For other scripts: run from `scripts/` directory or project root

### "File not found" errors
Check file locations with:
```bash
python scripts/config.py
```

### Path issues
All paths are now centralized. If you need custom paths, edit `scripts/config.py`

## Migration Notes

### What Changed?
- **Before**: Files scattered in root directory
- **After**: Organized into subdirectories

### What Stayed the Same?
- All functionality works exactly the same
- Same commands and workflows
- Database schema unchanged
- Model formats unchanged

### Backward Compatibility
Old scripts that haven't been updated yet will need to be run from their specific directories or updated to use the new config system.

## Need Help?

Check these files for more info:
- `README.md` - Full project documentation
- `docs/PROJECT_OVERVIEW.md` - Project methodology
- `docs/FINAL_MODEL_SUMMARY.md` - Model details
- `scripts/config.py` - All file paths

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Check configuration
python scripts/config.py

# 2. Process all XML data
cd scripts
python batch_processor.py

# 3. Engineer features with players
python feature_engineering_with_players.py

# 4. Train model
python train_xgboost_with_players.py

# 5. Run tournament simulation
cd ..
python run_simulation.py
```

All paths are handled automatically! 🎉
