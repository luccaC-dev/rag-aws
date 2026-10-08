variable "name" {
  type    = string
  default = "rag-demo"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image_tag" {
  type        = string
  description = "Tag of the image pushed to the ECR repository (immutable tags: use the git sha)."
}

variable "index_prefix" {
  type    = string
  default = "index"
}

variable "embed_model" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "chat_model" {
  type        = string
  description = "Bedrock model id or inference profile id enabled in this account and region."
}

variable "min_score" {
  type        = number
  default     = 0.2
  description = "Retrieval cutoff. Tune it with rag-eval for the chosen embedding model."
}

variable "price_in_per_1k" {
  type        = number
  default     = 0
  description = "USD per 1K input tokens of chat_model, from the Bedrock pricing page. 0 disables cost reporting."
}

variable "price_out_per_1k" {
  type    = number
  default = 0
}

variable "throttle_burst" {
  type    = number
  default = 5
}

variable "throttle_rate" {
  type    = number
  default = 2
}

variable "monthly_budget_usd" {
  type    = number
  default = 10
}

variable "alert_email" {
  type        = string
  description = "Receives the budget alert."
}
