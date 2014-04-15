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
package simulator.gpu.opencl.kernel.expression;

import java.util.List;

/**
 * @author mcopik
 *
 */
public class Expression implements KernelComponent
{
	String exprString;

	public Expression(String expr)
	{
		this.exprString = expr;
	}

	public Expression(Expression expr)
	{
		this.exprString = new String(expr.exprString);
	}

	public Expression add(String expr)
	{
		this.exprString += expr;
		return this;
	}

	public String getSource()
	{
		return exprString;
	}

	public String toString()
	{
		return exprString;
	}

	@Override
	public boolean hasIncludes()
	{
		return false;
	}

	@Override
	public boolean hasDeclaration()
	{
		return false;
	}

	@Override
	public List<Include> getIncludes()
	{
		return null;
	}

	@Override
	public KernelComponent getDeclaration()
	{
		return null;
	}

	@Override
	public void accept(VisitorInterface v)
	{
		v.visit(this);
	}
}