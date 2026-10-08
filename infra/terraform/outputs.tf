output "api_url" {
  value = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "index_s3_uri" {
  value = "s3://${aws_s3_bucket.index.bucket}/${var.index_prefix}"
}
