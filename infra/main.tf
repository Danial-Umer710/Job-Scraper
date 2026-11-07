terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40, < 7.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "table_name" {
  type    = string
  default = "job_scraper_state"
}

variable "github_repository" {
  description = "GitHub repo allowed to assume the CI role, as owner/name (e.g. daniyal/budapest-job-aggregator)."
  type        = string
}

variable "create_github_oidc_provider" {
  description = "Set to false if this AWS account already has the token.actions.githubusercontent.com OIDC provider."
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# State table
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "job_state" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  # Old records expire automatically so the table stays tiny (free tier).
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = {
    Project = "budapest-job-aggregator"
  }
}

# ---------------------------------------------------------------------------
# Keyless GitHub Actions access via OIDC (no long-lived AWS keys in GitHub)
# ---------------------------------------------------------------------------
resource "aws_iam_openid_connect_provider" "github" {
  count           = var.create_github_oidc_provider ? 1 : 0
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "job-aggregator-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

data "aws_iam_policy_document" "state_access" {
  statement {
    actions = [
      "dynamodb:BatchGetItem",
      "dynamodb:GetItem",
      "dynamodb:PutItem",
    ]
    resources = [aws_dynamodb_table.job_state.arn]
  }
}

resource "aws_iam_role_policy" "state_access" {
  name   = "job-aggregator-dynamodb"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.state_access.json
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.job_state.name
}

output "github_actions_role_arn" {
  description = "Store this as the AWS_ROLE_ARN secret in GitHub."
  value       = aws_iam_role.github_actions.arn
}
