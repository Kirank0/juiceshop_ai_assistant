/*
 * Copyright (c) 2014-2026 Bjoern Kimminich & the OWASP Juice Shop contributors.
 * SPDX-License-Identifier: MIT
 */

import { type ComponentFixture, TestBed } from '@angular/core/testing'
import { provideNoopAnimations } from '@angular/platform-browser/animations'
import { of, throwError } from 'rxjs'

import { AiAssistantWidgetComponent } from './ai-assistant-widget.component'
import { AiAssistantService } from '../Services/ai-assistant.service'

describe('AiAssistantWidgetComponent', () => {
  let component: AiAssistantWidgetComponent
  let fixture: ComponentFixture<AiAssistantWidgetComponent>
  let aiAssistantService: any

  beforeEach(async () => {
    aiAssistantService = {
      ask: vi.fn().mockName('AiAssistantService.ask')
    }
    aiAssistantService.ask.mockReturnValue(of({ reply: 'Apple Juice costs $1.99.', blocked: false }))

    await TestBed.configureTestingModule({
      imports: [AiAssistantWidgetComponent],
      providers: [
        { provide: AiAssistantService, useValue: aiAssistantService },
        provideNoopAnimations()
      ]
    }).compileComponents()

    fixture = TestBed.createComponent(AiAssistantWidgetComponent)
    component = fixture.componentInstance
    fixture.detectChanges()
  })

  it('should create', () => {
    expect(component).toBeTruthy()
  })

  it('should be closed initially and open on toggle', () => {
    expect(component.isOpen()).toBe(false)
    component.toggle()
    expect(component.isOpen()).toBe(true)
  })

  it('should send the message with previous history and append the reply', () => {
    component.draft = 'How much is Apple Juice?'
    component.send()

    expect(aiAssistantService.ask).toHaveBeenCalledWith('How much is Apple Juice?', [])
    const messages = component.messages()
    expect(messages[messages.length - 2]).toEqual({ role: 'user', content: 'How much is Apple Juice?' })
    expect(messages[messages.length - 1]).toEqual({ role: 'assistant', content: 'Apple Juice costs $1.99.' })
    expect(component.draft).toBe('')
    expect(component.isLoading()).toBe(false)

    component.draft = 'And the deluxe price?'
    component.send()
    expect(aiAssistantService.ask).toHaveBeenLastCalledWith('And the deluxe price?', [
      { role: 'user', content: 'How much is Apple Juice?' },
      { role: 'assistant', content: 'Apple Juice costs $1.99.' }
    ])
  })

  it('should not send empty messages', () => {
    component.draft = '   '
    component.send()
    expect(aiAssistantService.ask).not.toHaveBeenCalled()
  })

  it('should show an error when the assistant is unreachable', () => {
    aiAssistantService.ask.mockReturnValue(throwError(() => ({ status: 0 })))
    component.draft = 'Hello'
    component.send()
    expect(component.error()).toContain('not reachable')
    expect(component.isLoading()).toBe(false)
  })

  it('should show the error detail returned by the assistant', () => {
    aiAssistantService.ask.mockReturnValue(throwError(() => ({ status: 503, error: { detail: 'Juice Shop is down' } })))
    component.draft = 'Hello'
    component.send()
    expect(component.error()).toBe('Juice Shop is down')
  })
})
