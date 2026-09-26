output "function_name" {
  description = "Lambda function name."
  value       = aws_lambda_function.tools.function_name
}

output "function_arn" {
  description = "Lambda function ARN."
  value       = aws_lambda_function.tools.arn
}

output "function_url" {
  description = "HTTPS endpoint (AWS_IAM auth: requests must be SigV4-signed)."
  value       = aws_lambda_function_url.tools.function_url
}

output "role_arn" {
  description = "Execution role; it can only write the function's own log streams."
  value       = aws_iam_role.function.arn
}

output "artifact_bucket" {
  description = "Private bucket holding the code zip."
  value       = aws_s3_bucket.artifacts.bucket
}

output "invoker_policy_json" {
  description = "Identity policy to attach to a same-account role that should call the URL."
  value       = data.aws_iam_policy_document.invoker.json
}
