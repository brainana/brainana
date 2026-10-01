/*
 * Parameter Resolution Utility for Nextflow Workflows
 * 
 * Provides centralized parameter resolution with priority:
 * 1. Command-line arguments (params.*) - highest priority
 * 2. YAML config file (user-provided or defaults.yaml) - medium priority
 * 3. defaults.yaml - lowest priority
 * 
 * Usage:
 *   def paramResolver = load('workflows/param_resolver.groovy')
 *   def output_space = paramResolver.getParam('output_space', 'NMT2Sym:res-05')
 */

import groovy.yaml.YamlSlurper

// Wrap everything in a closure to ensure variable scope is captured
def createParamResolver = {
    // Mapping from CLI parameter names to YAML key paths
    // Only parameters that can be set via CLI and have different YAML paths need mapping
    def PARAM_MAPPING = [
        'output_space': 'template.output_space',
        'anat_only': 'general.anat_only',
        'subjects': 'bids_filtering.subjects',
        'sessions': 'bids_filtering.sessions',
        'tasks': 'bids_filtering.tasks',
        'runs': 'bids_filtering.runs'
    ]
    
    // Cache for loaded YAML config
    def yamlConfig = null
    def defaultsConfig = null
    def configLoaded = false

/**
 * Load YAML configuration file
 */
def loadYamlConfig = { configPath ->
    if (configPath == null || configPath.toString().trim().isEmpty()) {
        return [:]
    }
    
    def configFile = new File(configPath.toString())
    if (!configFile.exists()) {
        throw new FileNotFoundException("Config file not found: ${configPath}")
    }
    
    try {
        def yamlSlurper = new YamlSlurper()
        def config = yamlSlurper.parse(configFile)
        return config ?: [:]
    } catch (Exception e) {
        throw new RuntimeException("Failed to load YAML config from ${configPath}: ${e.message}", e)
    }
}

/**
 * Initialize the parameter resolver
 * Must be called before using getParam functions
 * Safe to call multiple times - will only load configs once
 */
def initialize = { params, projectDir ->
    // Skip if already initialized
    if (configLoaded) {
        return
    }
    
    // Load defaults.yaml
    def defaultsPath = "${projectDir}/src/nhp_mri_prep/config/defaults.yaml"
    try {
        defaultsConfig = loadYamlConfig(defaultsPath)
    } catch (Exception e) {
        throw new RuntimeException("Failed to load defaults.yaml: ${e.message}", e)
    }
    
    // Load user config file if provided
    def configFilePath = params.config_file ?: "${projectDir}/src/nhp_mri_prep/config/defaults.yaml"
    try {
        yamlConfig = loadYamlConfig(configFilePath)
    } catch (Exception e) {
        println "Warning: Failed to load config file ${configFilePath}, using defaults only: ${e.message}"
        yamlConfig = [:]
    }
    
    configLoaded = true
}

/**
 * Get nested value from dictionary using dot-separated key path
 */
def getNestedValue = { config, keyPath, defaultValue = null ->
    if (config == null || !(config instanceof Map)) {
        return defaultValue
    }
    
    def keys = keyPath.split('\\.')
    def value = config
    
    for (def key in keys) {
        if (value instanceof Map) {
            value = value.get(key)
            if (value == null) {
                return defaultValue
            }
        } else {
            return defaultValue
        }
    }
    
    return value != null ? value : defaultValue
}

/**
 * Return true when output_space is *intended* as a custom template file path.
 * A registered spec (e.g. "NMT2Sym:res-05") never contains a separator and never
 * ends in a NIfTI extension; a custom template is a path or a bare NIfTI filename.
 * String-shape check only (no filesystem access); mirrors is_custom_template_path()
 * in src/nhp_mri_prep/utils/templates.py and the _looks_like_path check in main.nf.
 * Returns true for a wrong-extension path (e.g. "/data/tpl.mgz") so downstream
 * resolution rejects it loudly rather than falling through to registered-spec lookup.
 */
def isCustomTemplatePath = { value ->
    if (value == null) {
        return false
    }
    def str = value.toString().trim()
    if (str.isEmpty()) {
        return false
    }
    return str.contains("/") || str.endsWith(".nii") || str.endsWith(".nii.gz")
}

/**
 * Validate output_space format
 * Valid formats: "T1w", "TEMPLATE_NAME", "TEMPLATE_NAME:DESCRIPTION",
 * or a custom template file path (contains '/' or ends in .nii/.nii.gz).
 * Custom template paths resolve to the literal 'template' space downstream;
 * mirror is_custom_template_path() in src/nhp_mri_prep/utils/templates.py.
 */
def validateOutputSpace = { value ->
    if (value == null || value.toString().trim().isEmpty()) {
        return false
    }

    def str = value.toString().trim()

    // Allow "T1w"
    if (str == "T1w") {
        return true
    }

    // Allow a custom template file path (validated for existence in Python at resolve time)
    if (isCustomTemplatePath(str)) {
        return true
    }

    // Check for TEMPLATE_NAME:DESCRIPTION format
    // Template name should be alphanumeric with underscores/hyphens
    // Description is optional and can contain alphanumeric, underscores, hyphens, colons
    def pattern = ~/^[a-zA-Z0-9_-]+(:[a-zA-Z0-9_:-]+)?$/
    return pattern.matcher(str).matches()
}

/**
 * Validate boolean value
 */
def validateBoolean = { value ->
    if (value == null) {
        return false
    }
    
    if (value instanceof Boolean) {
        return true
    }
    
    def str = value.toString().toLowerCase().trim()
    return str in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']
}

/**
 * Convert value to boolean
 */
def toBoolean = { value ->
    if (value instanceof Boolean) {
        return value
    }
    
    if (value == null) {
        return false
    }
    
    def str = value.toString().toLowerCase().trim()
    return str in ['true', '1', 'yes', 'on']
}

/**
 * Validate integer value
 */
def validateInteger = { value, min = null, max = null ->
    if (value == null) {
        return false
    }
    
    try {
        def intVal = value instanceof Integer ? value : Integer.parseInt(value.toString())
        if (min != null && intVal < min) {
            return false
        }
        if (max != null && intVal > max) {
            return false
        }
        return true
    } catch (NumberFormatException e) {
        return false
    }
}

/**
 * Convert value to integer
 */
def toInteger = { value, defaultVal = 0 ->
    if (value instanceof Integer) {
        return value
    }
    
    if (value == null) {
        return defaultVal
    }
    
    try {
        return Integer.parseInt(value.toString())
    } catch (NumberFormatException e) {
        return defaultVal
    }
}

/**
 * Validate float value
 */
def validateFloat = { value, min = null, max = null ->
    if (value == null) {
        return false
    }
    
    try {
        def floatVal = value instanceof Float || value instanceof Double ? 
            value as Double : Double.parseDouble(value.toString())
        if (min != null && floatVal < min) {
            return false
        }
        if (max != null && floatVal > max) {
            return false
        }
        return true
    } catch (NumberFormatException e) {
        return false
    }
}

/**
 * Convert value to float
 */
def toFloat = { value, defaultVal = 0.0 ->
    if (value instanceof Float || value instanceof Double) {
        return value as Double
    }
    
    if (value == null) {
        return defaultVal
    }
    
    try {
        return Double.parseDouble(value.toString())
    } catch (NumberFormatException e) {
        return defaultVal
    }
}

/**
 * Validate list value
 */
def validateList = { value ->
    if (value == null) {
        return false
    }
    
    // Accept List, array, or comma-separated string
    if (value instanceof List || value instanceof Object[]) {
        return true
    }

    // Nextflow turns a single numeric CLI value (--subjects 01, --runs 1) into a number
    if (value instanceof Number) {
        return true
    }
    
    if (value instanceof String) {
        // Empty string is not a valid list
        if (value.trim().isEmpty()) {
            return false
        }
        // Comma-separated string is valid
        return true
    }
    
    return false
}

/**
 * Convert value to list
 */
def toList = { value, defaultVal = [] ->
    if (value == null) {
        return defaultVal
    }
    
    if (value instanceof List) {
        return value
    }
    
    if (value instanceof Object[]) {
        return value.toList()
    }

    // A single numeric CLI value. Nextflow has already parsed it, so a leading
    // zero (--subjects 01) is gone here; BIDS discovery, which does the actual
    // filtering, receives the original text.
    if (value instanceof Number) {
        return [value.toString()]
    }
    
    if (value instanceof String) {
        if (value.trim().isEmpty()) {
            return defaultVal
        }
        // Split by comma and trim each element
        return value.split(',').collect { it.trim() }.findAll { !it.isEmpty() }
    }
    
    return defaultVal
}

/**
 * Resolve parameter with priority: CLI params → YAML config → defaults.yaml
 * 
 * @param paramName CLI parameter name (e.g., 'output_space')
 * @param defaultValue Default value if not found anywhere
 * @return Resolved parameter value
 */
def resolveParam = { params, paramName, defaultValue ->
    if (!configLoaded) {
        throw new IllegalStateException("Parameter resolver not initialized. Call initialize() first.")
    }
    
    // Priority 1: Check CLI parameter (params.*)
    // In Nextflow, if param is not set via CLI, it will be null (or default from nextflow.config)
    // Since we removed defaults from nextflow.config, null means not set via CLI
    def cliValue = params.get(paramName)
    if (cliValue != null && cliValue.toString().trim() != '') {
        return cliValue
    }
    
    // Priority 2: Check YAML config file
    def yamlKey = PARAM_MAPPING.get(paramName)
    if (yamlKey != null) {
        // Parameter has mapping, check YAML
        def yamlValue = getNestedValue(yamlConfig, yamlKey)
        if (yamlValue != null) {
            return yamlValue
        }
    } else {
        // Parameter doesn't have mapping, it's YAML-only
        // This shouldn't happen for CLI params, but handle gracefully
        println "Warning: Parameter '${paramName}' has no CLI mapping, checking YAML directly"
        def yamlValue = getNestedValue(yamlConfig, paramName)
        if (yamlValue != null) {
            return yamlValue
        }
    }
    
    // Priority 3: Check defaults.yaml
    if (yamlKey != null) {
        def defaultYamlValue = getNestedValue(defaultsConfig, yamlKey)
        if (defaultYamlValue != null) {
            return defaultYamlValue
        }
    }
    
    // Fallback to provided default
    return defaultValue
}

/**
 * Get parameter value (auto-detect type from default)
 * If defaultValue is null, reads from defaults.yaml
 */
def getParam = { params, paramName, defaultValue = null ->
    // If no default provided, get from defaults.yaml
    if (defaultValue == null) {
        def yamlKey = PARAM_MAPPING.get(paramName)
        if (yamlKey != null) {
            defaultValue = getNestedValue(defaultsConfig, yamlKey)
        }
        if (defaultValue == null) {
            throw new IllegalArgumentException("No default value provided for parameter '${paramName}' and not found in defaults.yaml")
        }
    }
    
    def value = resolveParam(params, paramName, defaultValue)
    
    // Auto-convert based on default value type
    if (defaultValue instanceof Boolean) {
        return toBoolean(value)
    } else if (defaultValue instanceof Integer) {
        return toInteger(value, defaultValue)
    } else if (defaultValue instanceof Float || defaultValue instanceof Double) {
        return toFloat(value, defaultValue)
    } else if (defaultValue instanceof List) {
        return toList(value, defaultValue)
    }
    
    return value != null ? value.toString() : defaultValue.toString()
}

/**
 * Get boolean parameter with validation
 * If defaultValue is null, reads from defaults.yaml
 */
def getParamBool = { params, paramName, defaultValue = null ->
    // If no default provided, get from defaults.yaml
    if (defaultValue == null) {
        def yamlKey = PARAM_MAPPING.get(paramName)
        if (yamlKey != null) {
            defaultValue = getNestedValue(defaultsConfig, yamlKey)
        }
        if (defaultValue == null) {
            throw new IllegalArgumentException("No default value provided for parameter '${paramName}' and not found in defaults.yaml")
        }
    }
    
    def value = resolveParam(params, paramName, defaultValue)
    
    if (!validateBoolean(value)) {
        throw new IllegalArgumentException("Invalid boolean value for parameter '${paramName}': ${value}. Expected true/false, 1/0, yes/no, on/off.")
    }
    
    return toBoolean(value)
}

/**
 * Get integer parameter with validation
 */
def getParamInt = { params, paramName, defaultValue, min = null, max = null ->
    def value = resolveParam(params, paramName, defaultValue)
    
    if (!validateInteger(value, min, max)) {
        def rangeStr = ""
        if (min != null && max != null) {
            rangeStr = " (range: ${min}-${max})"
        } else if (min != null) {
            rangeStr = " (min: ${min})"
        } else if (max != null) {
            rangeStr = " (max: ${max})"
        }
        throw new IllegalArgumentException("Invalid integer value for parameter '${paramName}': ${value}${rangeStr}")
    }
    
    return toInteger(value, defaultValue)
}

/**
 * Get float parameter with validation
 */
def getParamFloat = { params, paramName, defaultValue, min = null, max = null ->
    def value = resolveParam(params, paramName, defaultValue)
    
    if (!validateFloat(value, min, max)) {
        def rangeStr = ""
        if (min != null && max != null) {
            rangeStr = " (range: ${min}-${max})"
        } else if (min != null) {
            rangeStr = " (min: ${min})"
        } else if (max != null) {
            rangeStr = " (max: ${max})"
        }
        throw new IllegalArgumentException("Invalid float value for parameter '${paramName}': ${value}${rangeStr}")
    }
    
    return toFloat(value, defaultValue)
}

/**
 * Get list parameter with validation
 * If defaultValue is null, reads from defaults.yaml
 */
def getParamList = { params, paramName, defaultValue = null ->
    // If no default provided, get from defaults.yaml
    if (defaultValue == null) {
        def yamlKey = PARAM_MAPPING.get(paramName)
        if (yamlKey != null) {
            defaultValue = getNestedValue(defaultsConfig, yamlKey)
        }
        // If still null, use empty list as default for list parameters
        if (defaultValue == null) {
            defaultValue = []
        }
    }
    
    def value = resolveParam(params, paramName, defaultValue)
    
    // If value is null, return the default (empty list or from defaults)
    if (value == null) {
        return defaultValue instanceof List ? defaultValue : []
    }
    
    // Validate the value
    if (!validateList(value)) {
        throw new IllegalArgumentException("Invalid list value for parameter '${paramName}': ${value}. Expected list, array, or comma-separated string.")
    }
    
    // Convert to list and return
    return toList(value, defaultValue)
}

/**
 * Get string parameter with output_space validation
 * If defaultValue is null, reads from defaults.yaml
 */
def getParamOutputSpace = { params, paramName, defaultValue = null ->
    // If no default provided, get from defaults.yaml
    if (defaultValue == null) {
        def yamlKey = PARAM_MAPPING.get(paramName)
        if (yamlKey != null) {
            defaultValue = getNestedValue(defaultsConfig, yamlKey)
        }
        if (defaultValue == null) {
            throw new IllegalArgumentException("No default value provided for parameter '${paramName}' and not found in defaults.yaml")
        }
    }
    
    def value = resolveParam(params, paramName, defaultValue)
    
    if (!validateOutputSpace(value)) {
        throw new IllegalArgumentException("Invalid output_space format: '${value}'. Expected 'T1w', 'TEMPLATE_NAME', or 'TEMPLATE_NAME:DESCRIPTION' (e.g., 'NMT2Sym:res-05')")
    }
    
    return value.toString()
}

/**
 * Get YAML-only parameter (not available via CLI)
 * Priority: YAML config → defaults.yaml → default
 * If defaultValue is null, must exist in defaults.yaml
 */
def getYamlParam = { yamlKey, defaultValue = null ->
    if (!configLoaded) {
        throw new IllegalStateException("Parameter resolver not initialized. Call initialize() first.")
    }
    
    // Priority 1: Check YAML config file
    def yamlValue = getNestedValue(yamlConfig, yamlKey)
    if (yamlValue != null) {
        return yamlValue
    }
    
    // Priority 2: Check defaults.yaml
    def defaultYamlValue = getNestedValue(defaultsConfig, yamlKey)
    if (defaultYamlValue != null) {
        return defaultYamlValue
    }
    
    // If no default provided and not in defaults.yaml, error
    if (defaultValue == null) {
        throw new IllegalArgumentException("Parameter '${yamlKey}' not found in YAML config or defaults.yaml, and no default value provided")
    }
    
    // Fallback to provided default
    return defaultValue
}

/**
 * Get YAML boolean parameter
 * If defaultValue is null, reads from defaults.yaml
 */
def getYamlBool = { yamlKey, defaultValue = null ->
    def value = getYamlParam(yamlKey, defaultValue)
    
    // Get the actual default for error message
    def actualDefault = defaultValue
    if (actualDefault == null) {
        actualDefault = getNestedValue(defaultsConfig, yamlKey)
    }
    
    if (!validateBoolean(value)) {
        println "Warning: Invalid boolean value for YAML key '${yamlKey}': ${value}, using default: ${actualDefault}"
        return actualDefault instanceof Boolean ? actualDefault : toBoolean(actualDefault)
    }
    
    return toBoolean(value)
}

/**
 * Get YAML string parameter
 */
def getYamlString = { yamlKey, defaultValue ->
    def value = getYamlParam(yamlKey, defaultValue)
    return value != null ? value.toString() : defaultValue.toString()
}

/**
 * Get YAML integer parameter
 */
def getYamlInt = { yamlKey, defaultValue, min = null, max = null ->
    def value = getYamlParam(yamlKey, defaultValue)
    
    if (!validateInteger(value, min, max)) {
        def rangeStr = ""
        if (min != null && max != null) {
            rangeStr = " (range: ${min}-${max})"
        } else if (min != null) {
            rangeStr = " (min: ${min})"
        } else if (max != null) {
            rangeStr = " (max: ${max})"
        }
        println "Warning: Invalid integer value for YAML key '${yamlKey}': ${value}${rangeStr}, using default: ${defaultValue}"
        return defaultValue instanceof Integer ? defaultValue : toInteger(defaultValue, defaultValue)
    }
    
    return toInteger(value, defaultValue)
}

/**
 * Get YAML float parameter
 */
def getYamlFloat = { yamlKey, defaultValue, min = null, max = null ->
    def value = getYamlParam(yamlKey, defaultValue)
    
    if (!validateFloat(value, min, max)) {
        def rangeStr = ""
        if (min != null && max != null) {
            rangeStr = " (range: ${min}-${max})"
        } else if (min != null) {
            rangeStr = " (min: ${min})"
        } else if (max != null) {
            rangeStr = " (max: ${max})"
        }
        println "Warning: Invalid float value for YAML key '${yamlKey}': ${value}${rangeStr}, using default: ${defaultValue}"
        return defaultValue instanceof Float || defaultValue instanceof Double ? defaultValue : toFloat(defaultValue, defaultValue)
    }
    
    return toFloat(value, defaultValue)
}

/**
 * Get YAML list parameter
 */
def getYamlList = { yamlKey, defaultValue ->
    def value = getYamlParam(yamlKey, defaultValue)
    
    if (!validateList(value)) {
        println "Warning: Invalid list value for YAML key '${yamlKey}': ${value}, using default"
        return defaultValue instanceof List ? defaultValue : toList(defaultValue, defaultValue)
    }
    
    return toList(value, defaultValue)
}

/**
 * Parse general.gpu_device into [mode: 'auto'|'cpu'|'index', index: Integer].
 * Accepts the same spellings as the Python resolve_device(): auto, cuda, gpu,
 * cpu, -1 (int or quoted), N / "N" / "cuda:N" for a physical GPU index.
 */
def parseGpuDevice = { value ->
    def s = value == null ? 'auto' : value.toString().trim().toLowerCase()
    if (s in ['', 'auto', 'cuda', 'gpu']) {
        return [mode: 'auto', index: null]
    }
    if (s in ['cpu', '-1']) {
        return [mode: 'cpu', index: null]
    }
    def digits = s.startsWith('cuda:') ? s.substring(5) : s
    if (digits.isInteger() && digits.toInteger() >= 0) {
        return [mode: 'index', index: digits.toInteger()]
    }
    throw new IllegalArgumentException(
        "general.gpu_device must be auto, cpu, -1, a GPU index (0, 1, ...) or cuda:N; got '${value}'"
    )
}

/**
 * Whether workflows should schedule GPU processes on GPU tokens.
 * True when a GPU was detected (params.gpu_count, set in nextflow.config) and
 * general.gpu_device does not force CPU (-1 / "cpu").
 * Derived on demand rather than stored in params.use_gpu: a param declared in
 * nextflow.config is read-only to the script, so a runtime assignment is ignored.
 */
def resolveUseGpu = { params ->
    def spec = parseGpuDevice(getYamlParam('general.gpu_device', 'auto'))
    return spec.mode != 'cpu' && ((params.gpu_count ?: 0) > 0)
}

/**
 * Whether a registration step should take a GPU token: only FireANTs SyN uses the
 * GPU, so ANTs-only and rigid/affine registration stay CPU tasks and do not hold
 * a token another GPU step could use.
 */
def registrationUsesGpu = { params, List xfmKeys ->
    if (!resolveUseGpu(params)) {
        return false
    }
    def fireants = getYamlParam('registration.enable_fireants', true)
    if (!(fireants.toString().trim().toLowerCase() in ['true', 'yes', '1'])) {
        return false
    }
    return xfmKeys.any { key ->
        getYamlParam(key, 'syn').toString().trim().toLowerCase() == 'syn'
    }
}

/**
 * Physical GPU ids the token pool may hand out. Empty when GPU scheduling is off.
 * An explicit general.gpu_device index pins every GPU task to that one GPU.
 */
def resolveGpuIds = { params ->
    if (!resolveUseGpu(params)) {
        return []
    }
    def gpuCount = params.gpu_count ?: 0
    def spec = parseGpuDevice(getYamlParam('general.gpu_device', 'auto'))
    if (spec.mode == 'index') {
        if (spec.index >= gpuCount) {
            throw new IllegalArgumentException(
                "general.gpu_device: ${spec.index} but only ${gpuCount} GPU(s) detected (valid: 0..${gpuCount - 1})"
            )
        }
        return [spec.index]
    }
    return (0..<gpuCount).toList()
}

/**
 * Deep merge two maps (recursive)
 */
def deepMerge
deepMerge = { base, override ->
    def result = [:]
    if (base instanceof Map) {
        result.putAll(base)
    }
    
    if (override instanceof Map) {
        override.each { key, value ->
            if (result.containsKey(key) && result[key] instanceof Map && value instanceof Map) {
                result[key] = deepMerge.call(result[key], value)
            } else {
                result[key] = value
            }
        }
    }
    
    return result
}

/**
 * Set nested value in map using dot-separated key path
 */
def setNestedValue = { config, keyPath, value ->
    def keys = keyPath.split('\\.')
    def current = config
    
    // Navigate/create nested structure
    for (int i = 0; i < keys.length - 1; i++) {
        def key = keys[i]
        if (!current.containsKey(key) || !(current[key] instanceof Map)) {
            current[key] = [:]
        }
        current = current[key]
    }
    
    // Set the final value
    current[keys[keys.length - 1]] = value
}

/**
 * Generate effective config file by merging: CLI params → YAML config → defaults.yaml
 * Writes to output_dir/nextflow_reports/config.yaml
 */
def generateEffectiveConfig = { params, projectDir, outputDir ->
    if (!configLoaded) {
        throw new IllegalStateException("Parameter resolver not initialized. Call initialize() first.")
    }
    
    // Start with deep copy of defaults.yaml
    def effectiveConfig = deepMerge.call([:], defaultsConfig)
    
    // Merge user YAML config on top
    if (yamlConfig != null && !yamlConfig.isEmpty()) {
        effectiveConfig = deepMerge.call(effectiveConfig, yamlConfig)
    }
    
    // Override with CLI params (mapped to YAML keys)
    PARAM_MAPPING.each { cliParam, yamlKey ->
        def cliValue = params.get(cliParam)
        if (cliValue != null && cliValue.toString().trim() != '') {
            // Convert CLI value to appropriate type
            def value = cliValue
            if (cliValue instanceof String) {
                // Try to convert string to appropriate type based on default
                def defaultVal = getNestedValue(defaultsConfig, yamlKey)
                if (defaultVal instanceof Boolean) {
                    value = toBoolean(cliValue)
                } else if (defaultVal instanceof Integer || defaultVal instanceof Long) {
                    try {
                        value = Integer.parseInt(cliValue.toString())
                    } catch (NumberFormatException e) {
                        value = cliValue
                    }
                } else if (defaultVal instanceof Float || defaultVal instanceof Double) {
                    try {
                        value = Double.parseDouble(cliValue.toString())
                    } catch (NumberFormatException e) {
                        value = cliValue
                    }
                } else if (defaultVal instanceof List) {
                    value = toList(cliValue, defaultVal)
                }
            }
            setNestedValue(effectiveConfig, yamlKey, value)
        }
    }
    
    // Write effective config to file using Python (more reliable for YAML formatting)
    def configOutputPath = "${outputDir}/nextflow_reports/config.yaml"
    def configOutputFile = new File(configOutputPath)
    configOutputFile.parentFile.mkdirs()
    
    // Write config as JSON to temp file, then convert to YAML using Python
    def tempJsonFile = File.createTempFile("effective_config", ".json")
    tempJsonFile.deleteOnExit()
    
    // Convert Groovy map to JSON
    def jsonConfig = new groovy.json.JsonBuilder(effectiveConfig).toPrettyString()
    tempJsonFile.text = jsonConfig
    
    // Use Python to convert JSON to YAML, but only write if content changed
    // This preserves the file's mtime when content is unchanged, which is
    // critical for Nextflow's resume/caching to work correctly.
    def pythonScript = """
import yaml
import json
from pathlib import Path

# Read JSON config
with open('${tempJsonFile.absolutePath}', 'r') as f:
    config = json.load(f)

# Generate new YAML content
new_content = yaml.dump(config, default_flow_style=False, sort_keys=False, indent=2, allow_unicode=True)

# Check if file exists and has same content
output_path = Path('${configOutputPath}')
output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.exists():
    existing_content = output_path.read_text()
    if existing_content == new_content:
        # Content unchanged, skip writing to preserve mtime
        exit(0)

# Content changed or file doesn't exist, write new content
with open(output_path, 'w') as f:
    f.write(new_content)
"""
    
    // Execute Python script
    def proc = ["python3", "-c", pythonScript].execute()
    def output = new StringBuffer()
    def error = new StringBuffer()
    proc.consumeProcessOutput(output, error)
    proc.waitFor()
    
    if (proc.exitValue() != 0) {
        throw new RuntimeException("Failed to write effective config file: ${configOutputPath}. Error: ${error.toString()}")
    }
    
    return configOutputPath
}

    // Return resolver functions as closures (to capture variable scope)
    return [
        initialize: { params, projectDir -> initialize(params, projectDir) },
        generateEffectiveConfig: { params, projectDir, outputDir -> generateEffectiveConfig(params, projectDir, outputDir) },
        getParam: { params, paramName, defaultValue = null -> getParam(params, paramName, defaultValue) },
        getParamBool: { params, paramName, defaultValue = null -> getParamBool(params, paramName, defaultValue) },
        getParamInt: { params, paramName, defaultValue, min = null, max = null -> getParamInt(params, paramName, defaultValue, min, max) },
        getParamFloat: { params, paramName, defaultValue, min = null, max = null -> getParamFloat(params, paramName, defaultValue, min, max) },
        getParamList: { params, paramName, defaultValue -> getParamList(params, paramName, defaultValue) },
        getParamOutputSpace: { params, paramName, defaultValue = null -> getParamOutputSpace(params, paramName, defaultValue) },
        isCustomTemplatePath: { value -> isCustomTemplatePath(value) },
        getYamlParam: { yamlKey, defaultValue = null -> getYamlParam(yamlKey, defaultValue) },
        getYamlBool: { yamlKey, defaultValue = null -> getYamlBool(yamlKey, defaultValue) },
        getYamlString: { yamlKey, defaultValue = null -> getYamlString(yamlKey, defaultValue) },
        getYamlInt: { yamlKey, defaultValue = null, min = null, max = null -> getYamlInt(yamlKey, defaultValue, min, max) },
        getYamlFloat: { yamlKey, defaultValue = null, min = null, max = null -> getYamlFloat(yamlKey, defaultValue, min, max) },
        getYamlList: { yamlKey, defaultValue = null -> getYamlList(yamlKey, defaultValue) },
        resolveUseGpu: { params -> resolveUseGpu(params) },
        resolveGpuIds: { params -> resolveGpuIds(params) },
        registrationUsesGpu: { params, xfmKeys -> registrationUsesGpu(params, xfmKeys) }
    ]
}

// Execute the closure and return the resolver
return createParamResolver()
