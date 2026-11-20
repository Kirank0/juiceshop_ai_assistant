/*
 * Copyright (c) 2014-2026 Bjoern Kimminich & the OWASP Juice Shop contributors.
 * SPDX-License-Identifier: MIT
 */

import { Component, ChangeDetectionStrategy, ElementRef, inject, signal, viewChild } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { MatButtonModule } from '@angular/material/button'
import { MatCardModule } from '@angular/material/card'
import { MatIconModule } from '@angular/material/icon'
import { MatTooltipModule } from '@angular/material/tooltip'
import { type HttpErrorResponse } from '@angular/common/http'
import { AiAssistantService, type AiAssistantMessage } from '../Services/ai-assistant.service'

const MAX_HISTORY_MESSAGES = 20

@Component({
  selector: 'app-ai-assistant-widget',
  templateUrl: './ai-assistant-widget.component.html',
  styleUrls: ['./ai-assistant-widget.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, MatButtonModule, MatCardModule, MatIconModule, MatTooltipModule]
})
export class AiAssistantWidgetComponent {
  private readonly aiAssistantService = inject(AiAssistantService)
  private readonly messageList = viewChild<ElementRef<HTMLElement>>('messageList')

  readonly isOpen = signal(false)
  readonly isLoading = signal(false)
  readonly error = signal<string | null>(null)
  readonly messages = signal<AiAssistantMessage[]>([
    { role: 'assistant', content: 'Hi! I can help you with our products and their prices. What are you looking for?' }
  ])

  draft = ''

  toggle () {
    this.isOpen.update(open => !open)
    this.scrollToBottom()
  }

  send () {
    const message = this.draft.trim()
    if (!message || this.isLoading()) {
      return
    }
    const history = this.messages().slice(1).slice(-MAX_HISTORY_MESSAGES)
    this.messages.update(messages => [...messages, { role: 'user', content: message }])
    this.draft = ''
    this.error.set(null)
    this.isLoading.set(true)
    this.scrollToBottom()

    this.aiAssistantService.ask(message, history).subscribe({
      next: (response) => {
        this.messages.update(messages => [...messages, { role: 'assistant', content: response.reply }])
        this.isLoading.set(false)
        this.scrollToBottom()
      },
      error: (err: HttpErrorResponse) => {
        this.error.set(err.status === 0
          ? 'The assistant is not reachable right now. Please try again later.'
          : (err.error?.detail ?? 'Something went wrong. Please try again.'))
        this.isLoading.set(false)
        this.scrollToBottom()
      }
    })
  }

  onKeydown (event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      this.send()
    }
  }

  private scrollToBottom () {
    setTimeout(() => {
      const element = this.messageList()?.nativeElement
      if (element) {
        element.scrollTop = element.scrollHeight
      }
    })
  }
}
