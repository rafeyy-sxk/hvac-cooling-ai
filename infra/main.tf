# The cooler tools (psychrometrics, physics, surrogate, sizing) as one AWS Lambda function behind a
# function URL with AWS_IAM auth. Validated with `terraform validate`; not deployed.
#
# Build the package first:  infra/build_lambda.sh   (about 61 MB zipped, so it goes through S3:
# direct zip uploads to Lambda are limited to 50 MB).

locals {
  handler        = "hvac_cooling_ai.lambda_handler.handler"
  log_group_name = "/aws/lambda/${var.function_name}"
  package_sha256 = filesha256(var.package_path)
}

# ---- code artifact bucket (private, encrypted, versioned) ----

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${var.function_name}-artifacts-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_object" "package" {
  bucket      = aws_s3_bucket.artifacts.id
  key         = "lambda/${local.package_sha256}.zip" # content-addressed: new code, new key
  source      = var.package_path
  source_hash = local.package_sha256

  depends_on = [aws_s3_bucket_server_side_encryption_configuration.artifacts]
}

# ---- logs and execution role (least privilege) ----

resource "aws_cloudwatch_log_group" "function" {
  name              = local.log_group_name
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "assume_lambda" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "function" {
  name               = "${var.function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.assume_lambda.json
}

# The function only writes its own log streams. Terraform creates the log group, so the role needs no
# logs:CreateLogGroup, and no S3 access: Lambda fetches the code with the deployer's credentials.
data "aws_iam_policy_document" "function" {
  statement {
    sid       = "WriteOwnLogStreams"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.function.arn}:*"]
  }
}

resource "aws_iam_role_policy" "function" {
  name   = "${var.function_name}-logs"
  role   = aws_iam_role.function.id
  policy = data.aws_iam_policy_document.function.json
}

# ---- the function and its URL ----

resource "aws_lambda_function" "tools" {
  function_name                  = var.function_name
  description                    = "Indirect evaporative cooler tools: psychrometrics, physics model, ML surrogate, sizing"
  role                           = aws_iam_role.function.arn
  runtime                        = "python3.12"
  architectures                  = ["arm64"]
  handler                        = local.handler
  s3_bucket                      = aws_s3_object.package.bucket
  s3_key                         = aws_s3_object.package.key
  source_code_hash               = filebase64sha256(var.package_path)
  memory_size                    = var.memory_mb
  timeout                        = var.timeout_seconds
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = {
      # Lambda hides the physical core count; without this joblib warns on every cold start.
      LOKY_MAX_CPU_COUNT = "1"
    }
  }

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.function.name
  }

  depends_on = [aws_iam_role_policy.function]
}

resource "aws_lambda_function_url" "tools" {
  function_name      = aws_lambda_function.tools.function_name
  authorization_type = "AWS_IAM"
}

# Since October 2025 a function URL caller needs both lambda:InvokeFunctionUrl and
# lambda:InvokeFunction; the second is limited to calls that arrive through the URL.
resource "aws_lambda_permission" "url_invoke" {
  for_each = toset(var.invoker_principal_arns)

  statement_id           = "FunctionUrl-${substr(sha1(each.value), 0, 12)}"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.tools.function_name
  principal              = each.value
  function_url_auth_type = "AWS_IAM"
}

resource "aws_lambda_permission" "url_invoke_function" {
  for_each = toset(var.invoker_principal_arns)

  statement_id             = "FunctionUrlInvoke-${substr(sha1(each.value), 0, 12)}"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.tools.function_name
  principal                = each.value
  invoked_via_function_url = true
}

# An identity policy a same-account caller can attach to call the URL and nothing else.
data "aws_iam_policy_document" "invoker" {
  statement {
    sid       = "CallToolsUrl"
    actions   = ["lambda:InvokeFunctionUrl"]
    resources = [aws_lambda_function.tools.arn]

    condition {
      test     = "StringEquals"
      variable = "lambda:FunctionUrlAuthType"
      values   = ["AWS_IAM"]
    }
  }

  statement {
    sid       = "InvokeViaUrlOnly"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.tools.arn]

    condition {
      test     = "Bool"
      variable = "lambda:InvokedViaFunctionUrl"
      values   = ["true"]
    }
  }
}
