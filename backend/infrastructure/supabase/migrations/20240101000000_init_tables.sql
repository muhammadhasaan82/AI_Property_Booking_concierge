-- Create FAQ table
CREATE TABLE IF NOT EXISTS public.faqs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster FAQ lookups
CREATE INDEX IF NOT EXISTS idx_faqs_question ON public.faqs USING gin(to_tsvector('english', question));

-- Insert some sample FAQ data
INSERT INTO public.faqs (question, answer) VALUES
('What is the refund policy?', 'The refund policy for short-stay bookings is as follows: 15+ days before check-in: 90% refund, 7-14 days: 50% refund, 4-6 days: 40% refund, under 3 days: 25% refund. Refunds are processed within 7-10 business days.'),
('What is the check-in time?', 'Standard check-in time is 3:00 PM. Early check-in may be available upon request and subject to availability.'),
('What is the check-out time?', 'Standard check-out time is 11:00 AM. Late check-out may be available upon request and subject to availability.'),
('Can I cancel my booking?', 'Yes, you can cancel your booking. The refund amount depends on how far in advance you cancel, as per our refund policy.'),
('What payment methods do you accept?', 'We accept all major credit cards, PayPal, and bank transfers. Payment is required at the time of booking.'),
('Is WiFi included?', 'Yes, complimentary WiFi is included in all our properties.'),
('Are pets allowed?', 'Pet policies vary by property. Please check the specific property details or contact us for pet-friendly options.'),
('What amenities are included?', 'Most properties include WiFi, kitchen facilities, bathroom essentials, and basic cleaning supplies. Specific amenities vary by property.'),
('How do I get the keys?', 'Key collection details will be provided after booking confirmation. This may include keyless entry codes, key pickup locations, or host meet-and-greet arrangements.'),
('What if I have an emergency?', 'Emergency contact information will be provided with your booking confirmation. We have 24/7 support for urgent matters.');

-- Enable Row Level Security
ALTER TABLE public.faqs ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access
CREATE POLICY "Allow public read access to FAQs" ON public.faqs
    FOR SELECT USING (true);

