//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.automaton.command;

import simulator.gpu.automaton.Guard;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.automaton.update.Update;

public class Command implements CommandInterface
{
	private final Guard guard;
	private final Update update;
	private final Rate rate;

	private Command(Guard guard, Update update, Rate rate)
	{
		this.guard = guard;
		this.update = update;
		this.rate = rate;
	}

	static public Command createCommandDTMC(Guard guard, Update update)
	{
		return new Command(guard, update, new Rate(1));
	}

	static public Command createCommandCTMC(Guard guard, Update update)
	{
		Rate rate = new Rate(0);
		for (int i = 0; i < update.getActionsNumber(); ++i) {
			rate.addRate(update.getRate(i));
		}
		return new Command(guard, update, rate);
	}

	@Override
	public Guard getGuard()
	{
		return guard;
	}

	@Override
	public Update getUpdate()
	{
		return update;
	}

	@Override
	public boolean isSynchronized()
	{
		return false;
	}

	@Override
	public String toString()
	{
		StringBuilder builder = new StringBuilder();
		builder.append(guard).append(" -> ").append(update).append("\n");
		return builder.toString();
	}

	@Override
	public Rate getRateSum()
	{
		return rate;
	}
}