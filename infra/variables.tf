variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "function_name" {
  description = "Lambda function name; also prefixes the role, log group and artifact bucket."
  type        = string
  default     = "hvac-cooling-tools"

  validation {
    # the artifact bucket_prefix is "<name>-artifacts-" and S3 allows at most 37 characters of prefix
    condition     = can(regex("^[a-z0-9-]{1,26}$", var.function_name))
    error_message = "function_name must be 1-26 characters of lowercase letters, digits and hyphens."
  }
}

variable "package_path" {
  description = "Path to the zip built by infra/build_lambda.sh."
  type        = string
  default     = "build/lambda.zip"
}

variable "memory_mb" {
  description = "Lambda memory in MB (CPU scales with it)."
  type        = number
  default     = 1024

  validation {
    condition     = var.memory_mb >= 512 && var.memory_mb <= 10240
    error_message = "memory_mb must be between 512 and 10240 (numpy, scipy and scikit-learn need at least 512)."
  }
}

variable "timeout_seconds" {
  description = "Per-invocation timeout. One physics run takes about 0.4 s on a laptop; a cold start loads scikit-learn."
  type        = number
  default     = 30

  validation {
    condition     = var.timeout_seconds >= 3 && var.timeout_seconds <= 900
    error_message = "timeout_seconds must be between 3 and 900."
  }
}

variable "reserved_concurrency" {
  description = "Hard cap on concurrent executions, so a runaway caller cannot run up the bill."
  type        = number
  default     = 5

  validation {
    condition     = var.reserved_concurrency >= 1
    error_message = "reserved_concurrency must be at least 1 (0 would disable the function)."
  }
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the function."
  type        = number
  default     = 14
}

variable "invoker_principal_arns" {
  description = <<-EOT
    IAM principals (role or user ARNs, or account ids) allowed to call the function URL through the
    function's resource policy. The URL uses AWS_IAM auth: it is never public. Same-account callers can
    instead attach the policy in the invoker_policy_json output to their own role.
  EOT
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default = {
    project = "hvac-cooling-ai"
  }
}
