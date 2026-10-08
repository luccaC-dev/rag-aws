terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { project = var.name }
  }
}

data "aws_caller_identity" "current" {}

# --- storage ----------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_s3_bucket" "index" {
  bucket_prefix = "${var.name}-index-"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- lambda -----------------------------------------------------------------

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 14
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "app" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.app.arn}:*"]
  }
  statement {
    sid       = "ReadIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.index.arn}/${var.index_prefix}/*"]
  }
  statement {
    sid     = "InvokeModels"
    actions = ["bedrock:InvokeModel"]
    # Foundation models plus inference profiles (some models are only callable through one).
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "app" {
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app.json
}

resource "aws_lambda_function" "app" {
  function_name = var.name
  role          = aws_iam_role.app.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  memory_size   = 1024
  timeout       = 30
  architectures = ["x86_64"]

  environment {
    variables = {
      RAG_PROVIDER         = "bedrock"
      RAG_INDEX_PATH       = "/tmp/index"
      RAG_INDEX_S3_URI     = "s3://${aws_s3_bucket.index.bucket}/${var.index_prefix}"
      RAG_MIN_SCORE        = tostring(var.min_score)
      BEDROCK_EMBED_MODEL  = var.embed_model
      BEDROCK_CHAT_MODEL   = var.chat_model
      RAG_PRICE_IN_PER_1K  = tostring(var.price_in_per_1k)
      RAG_PRICE_OUT_PER_1K = tostring(var.price_out_per_1k)
    }
  }

  depends_on = [aws_cloudwatch_log_group.app, aws_iam_role_policy.app]
}

# --- http api ---------------------------------------------------------------

resource "aws_apigatewayv2_api" "http" {
  name          = var.name
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.app.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
  # Throttling caps model spend if the public URL gets hammered.
  default_route_settings {
    throttling_burst_limit = var.throttle_burst
    throttling_rate_limit  = var.throttle_rate
  }
}

resource "aws_lambda_permission" "api" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# --- cost guardrail ---------------------------------------------------------

resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
